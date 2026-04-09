import ast
import json
import os
import random
import re
import signal
import subprocess
import sys
import time
import uuid

import wandb

# Track the currently running child so termination signals can be propagated.
_ACTIVE_CHILD = None

SWEEP_ENV_KEYS = {
    "WANDB_SWEEP_ID",
    "WANDB_SWEEP_PARAM_PATH",
    "WANDB_AGENT_ID",
    "WANDB_RUN_ID",
}

ALLOWED_ALGOS = {"happo", "mappo", "mappo_t", "ippo", "coma", "mappo_vd"}
MAX_MACA_BASE_SEED = (2**32 - 1) // 50000 - 2

UNSUPPORTED_DISSC_KEYS = {
    # These are in DISSC PPO-family sweeps but have no direct on-policy equivalent in MACA.
    "standardise_rewards",
    "q_nstep",
    "target_update_interval_or_tau",
}

def _coerce_bool(value, key_name):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    raise ValueError(
        f"Sweep key '{key_name}' expects a boolean value, got {value!r}."
    )


def _parse_value(raw):
    """Parse a CLI value token from wandb agent into Python types."""
    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(raw)
        except Exception:
            pass
    if raw == "True":
        return True
    if raw == "False":
        return False
    return raw


def _serialize_value(value):
    """Serialize values into MACA CLI-compatible strings."""
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return "[" + ",".join(_serialize_value(v) for v in value) + "]"
    if isinstance(value, tuple):
        return "(" + ",".join(_serialize_value(v) for v in value) + ")"
    return str(value)


def parse_sweep_cli(argv):
    """Parse wandb agent CLI args of the form --key=value."""
    parsed = {}
    for token in argv:
        if not token.startswith("--"):
            continue
        content = token[2:]
        if "=" not in content:
            key, value = content, True
        else:
            key, raw_value = content.split("=", 1)
            value = _parse_value(raw_value)
        if key in parsed:
            raise ValueError(f"Duplicate sweep key '{key}' received by wrapper.")
        parsed[key] = value
    return parsed


def _extract_meta(params):
    """Extract sweep metadata/entrypoint controls from raw params."""
    params = dict(params)
    env_config = params.pop("env-config", None)
    config_name = params.pop("config", None)
    algo = params.pop("algo", None)
    exp_name = params.pop("name", None)
    env_name = params.pop("env", None)

    if env_name is None:
        if env_config in (None, "gymma"):
            env_name = "gym"
        else:
            raise ValueError(
                f"Unsupported env-config '{env_config}'. "
                "This wrapper currently supports gymma-style sweeps only."
            )

    if algo is None:
        if config_name in ALLOWED_ALGOS:
            algo = config_name
        else:
            raise ValueError(
                "Unable to infer MACA algo from sweep key 'config'. "
                "Set explicit sweep parameter 'algo' to one of: "
                + ", ".join(sorted(ALLOWED_ALGOS))
                + "."
            )
    elif algo not in ALLOWED_ALGOS:
        raise ValueError(
            f"Unsupported algo '{algo}'. Expected one of: {sorted(ALLOWED_ALGOS)}"
        )

    if exp_name is None:
        exp_name = config_name if config_name else algo

    return params, {
        "env": env_name,
        "algo": algo,
        "exp_name": exp_name,
        "config_name": config_name,
    }


def translate_params(raw_params):
    """
    Translate DISSCv2 sweep keys to MACA CLI override keys.

    Returns:
        meta: dict with main CLI args.
        overrides: dict of MACA override keys for examples.train.
        warnings: list of warning strings.
    """
    params, meta = _extract_meta(raw_params)
    warnings = []
    overrides = {}

    for key in sorted(UNSUPPORTED_DISSC_KEYS):
        if key in params:
            raise ValueError(
                f"Sweep key '{key}' is not supported for MACA on-policy sweeps "
                "(no equivalent semantics). Remove it from the sweep config."
            )

    if "save_model" in params:
        save_model = _coerce_bool(params.pop("save_model"), "save_model")
        if save_model:
            raise ValueError(
                "Sweep key 'save_model=True' is not supported by this wrapper. "
                "MACA always writes a final checkpoint."
            )
        warnings.append(
            "Ignoring non-equivalent key 'save_model=False' (MACA always saves final checkpoint)."
        )

    # DISSC env-key style -> MACA gym args
    if "env_args.key" in params:
        overrides["scenario"] = params.pop("env_args.key")
    if "env_args.n_agents" in params:
        overrides["n_agents"] = int(params.pop("env_args.n_agents"))
    if "env_args.time_limit" in params:
        overrides["episode_length"] = int(params.pop("env_args.time_limit"))

    # DISSC global training length -> MACA
    if "t_max" in params:
        overrides["num_env_steps"] = int(params.pop("t_max"))

    # Hyperparameter mappings for mappo_lbf-style parity
    if "hidden_dim" in params:
        dim = int(params.pop("hidden_dim"))
        overrides["model.hidden_sizes"] = [dim, dim]
    if "lr" in params:
        lr = float(params.pop("lr"))
        overrides["model.lr"] = lr
        overrides["model.critic_lr"] = lr
    if "use_rnn" in params:
        use_rnn = _coerce_bool(params.pop("use_rnn"), "use_rnn")
        overrides["model.use_recurrent_policy"] = use_rnn
        overrides["model.use_naive_recurrent_policy"] = False
    if "entropy_coef" in params:
        overrides["algo.entropy_coef"] = float(params.pop("entropy_coef"))
    if "epochs" in params:
        epochs = int(params.pop("epochs"))
        overrides["algo.ppo_epoch"] = epochs
        overrides["algo.critic_epoch"] = epochs
    if "eps_clip" in params:
        overrides["algo.clip_param"] = float(params.pop("eps_clip"))

    # Pass through any already-MACA-compatible keys.
    for key, value in params.items():
        overrides[key] = value

    return meta, overrides, warnings


def build_child_cmd(meta, overrides, seed, wandb_project, wandb_entity, wandb_mode):
    """Build one child training command."""
    child_exp_name = f"{meta['exp_name']}_seed{seed}"
    cmd = [
        sys.executable,
        "-m",
        "examples.train",
        "--algo",
        meta["algo"],
        "--env",
        meta["env"],
        "--exp_name",
        child_exp_name,
        "--algo_args.seed.seed={}".format(seed),
        "--algo_args.logger.use_wandb=True",
        f"--algo_args.logger.wandb_project={wandb_project}",
        f"--algo_args.logger.wandb_entity={wandb_entity}",
        f"--algo_args.logger.wandb_mode={wandb_mode}",
    ]

    for key, value in overrides.items():
        cmd.append(f"--{key}={_serialize_value(value)}")

    return cmd


def run_seed(seed, cmd, clean_env, run_name):
    """Run one seed subprocess and return (run_id, return_code, log_path)."""
    global _ACTIVE_CHILD
    run_id = uuid.uuid4().hex[:8]
    env = dict(clean_env)
    env["WANDB_RUN_ID"] = run_id

    log_root = os.environ.get("SWEEP_LOG_DIR", "results/wandb_wrapper_logs")
    log_dir = os.path.join(log_root, run_name)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"seed_{seed}_{run_id}.log")

    print(f"[maca_sweep] starting seed={seed}, wandb_run_id={run_id}")
    with open(log_path, "w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            cmd,
            env=env,
            start_new_session=True,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        _ACTIVE_CHILD = proc
        proc.wait()
        _ACTIVE_CHILD = None
    print(f"[maca_sweep] seed={seed} finished with exit code {proc.returncode}")
    print(f"[maca_sweep] seed={seed} log={log_path}")
    return run_id, proc.returncode, log_path


def fetch_run_metric(api, entity, project, run_id, metric_name, max_retries=8, delay=8):
    """Fetch metric from run summary with retries to handle propagation delay."""
    if api is None:
        return None
    for attempt in range(max_retries):
        try:
            run = api.run(f"{entity}/{project}/{run_id}")
            if metric_name in run.summary:
                value = run.summary[metric_name]
                print(f"[maca_sweep] run={run_id} {metric_name}={value}")
                return value
            print(
                f"[maca_sweep] run={run_id} missing '{metric_name}' in summary "
                f"(attempt {attempt + 1}/{max_retries})"
            )
        except Exception as exc:
            print(
                f"[maca_sweep] run={run_id} metric fetch error "
                f"(attempt {attempt + 1}/{max_retries}): {exc}"
            )
        if attempt < max_retries - 1:
            time.sleep(delay)
    return None


def parse_metric_from_log(log_path, metric_name):
    """Fallback metric parser from child training logs."""
    if not log_path or not os.path.exists(log_path):
        return None

    pattern = re.compile(
        rf"{re.escape(metric_name)}:\s*"
        r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
    )
    try:
        with open(log_path, "r", encoding="utf-8") as file:
            content = file.read()
    except OSError:
        return None

    matches = pattern.findall(content)
    if not matches:
        return None
    return float(matches[-1])


def get_sweep_metric_name(api, entity, project, sweep_id):
    """Read metric name from sweep config via wandb API."""
    if api is None:
        return None
    if not sweep_id:
        return None
    try:
        sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
        metric_name = sweep.config.get("metric", {}).get("name")
        return metric_name
    except Exception as exc:
        print(f"[maca_sweep] warning: failed reading sweep metric from API: {exc}")
        return None


def _handle_signal(signum, _frame):
    """Propagate SIGTERM/SIGINT to active child process group."""
    if _ACTIVE_CHILD and _ACTIVE_CHILD.poll() is None:
        print(f"[maca_sweep] received signal {signum}, terminating child {_ACTIVE_CHILD.pid}")
        try:
            os.killpg(os.getpgid(_ACTIVE_CHILD.pid), signal.SIGTERM)
        except OSError:
            _ACTIVE_CHILD.kill()
    wandb.finish(exit_code=1)
    sys.exit(1)


signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT, _handle_signal)


def main():
    raw_params = parse_sweep_cli(sys.argv[1:])
    meta, overrides, warnings = translate_params(raw_params)

    for warning in warnings:
        print(f"[maca_sweep] warning: {warning}")

    num_seeds = int(os.environ.get("SWEEP_NUM_SEEDS", "3"))
    seeds = [random.randint(0, MAX_MACA_BASE_SEED) for _ in range(num_seeds)]

    hparam_tag = uuid.uuid4().hex[:6]
    run_name = f"{meta['exp_name']}_{hparam_tag}"

    wrapper_wandb_mode = os.environ.get("SWEEP_WRAPPER_WANDB_MODE", "online")
    sweep_run = wandb.init(
        config={
            "raw_params": raw_params,
            "translated_overrides": overrides,
            "meta": meta,
            "num_seeds": num_seeds,
            "seeds": seeds,
        },
        name=f"{run_name}-sweep",
        tags=["sweep_summary", "maca_multi_seed"],
        mode=wrapper_wandb_mode,
    )

    entity = sweep_run.entity
    project = sweep_run.project
    sweep_id = os.environ.get("WANDB_SWEEP_ID", "")
    child_wandb_mode = os.environ.get("SWEEP_CHILD_WANDB_MODE", "online")

    clean_env = {k: v for k, v in os.environ.items() if k not in SWEEP_ENV_KEYS}
    clean_env["WANDB_NAME"] = run_name

    print(
        f"[maca_sweep] sweep-run={sweep_run.id} project={project} entity={entity} "
        f"algo={meta['algo']} env={meta['env']} seeds={seeds}"
    )

    run_ids = []
    run_logs = {}
    for seed in seeds:
        cmd = build_child_cmd(meta, overrides, seed, project, entity, child_wandb_mode)
        run_id, rc, log_path = run_seed(seed, cmd, clean_env, run_name)
        if rc != 0:
            print(f"[maca_sweep] abort: child seed={seed} failed with exit code {rc}")
            wandb.finish(exit_code=rc)
            sys.exit(rc)
        run_ids.append(run_id)
        run_logs[run_id] = log_path

    api = None
    try:
        api = wandb.Api()
    except Exception as exc:
        print(
            "[maca_sweep] warning: wandb.Api unavailable; "
            f"falling back to local log metric parsing. ({exc})"
        )
    metric_name = get_sweep_metric_name(api, entity, project, sweep_id)
    if not metric_name:
        metric_name = os.environ.get("SWEEP_METRIC_NAME", "eval_episode_rewards_mean")
        print(f"[maca_sweep] using fallback metric name: {metric_name}")
    else:
        print(f"[maca_sweep] sweep metric name: {metric_name}")

    enable_log_fallback = os.environ.get("SWEEP_ENABLE_LOG_FALLBACK", "1") != "0"
    values = []
    missing_run_ids = []
    for run_id in run_ids:
        val = fetch_run_metric(api, entity, project, run_id, metric_name)
        if val is None and enable_log_fallback:
            val = parse_metric_from_log(run_logs.get(run_id), metric_name)
            if val is not None:
                print(
                    f"[maca_sweep] run={run_id} "
                    f"{metric_name}={val} (log fallback)"
                )
        if val is not None:
            values.append(float(val))
        else:
            missing_run_ids.append(run_id)

    if missing_run_ids:
        print(
            f"[maca_sweep] error: missing metric '{metric_name}' for child runs: "
            + ", ".join(missing_run_ids)
        )
        wandb.finish(exit_code=1)
        sys.exit(1)

    if not values:
        print(f"[maca_sweep] error: unable to fetch '{metric_name}' from any child run.")
        wandb.finish(exit_code=1)
        sys.exit(1)

    avg_value = sum(values) / len(values)
    print(
        f"[maca_sweep] avg {metric_name} across {len(values)} seeds: {avg_value}"
    )

    log_data = {metric_name: avg_value}
    for idx, (seed, value) in enumerate(zip(seeds, values)):
        log_data[f"{metric_name}_seed{idx}"] = value
        log_data[f"seed_{idx}"] = seed
    wandb.log(log_data)
    wandb.config.update({"seed_run_ids": run_ids})
    wandb.finish()


if __name__ == "__main__":
    main()
