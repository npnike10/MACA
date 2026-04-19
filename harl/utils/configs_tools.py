"""Tools for loading and updating configs."""
import time
import os
import json
import yaml
import torch as th
from uu import Error
import datetime


def get_defaults_yaml_args(algo, env):
    """Load config file for user-specified algo and env.
    Args:
        algo: (str) Algorithm name.
        env: (str) Environment name.
    Returns:
        algo_args: (dict) Algorithm config.
        env_args: (dict) Environment config.
    """
    base_path = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
    algo_cfg_path = os.path.join(base_path, "configs", "algos_cfgs", f"{algo}.yaml")
    env_cfg_path = os.path.join(base_path, "configs", "envs_cfgs", f"{env}.yaml")

    with open(algo_cfg_path, "r", encoding="utf-8") as file:
        algo_args = yaml.load(file, Loader=yaml.FullLoader)
    with open(env_cfg_path, "r", encoding="utf-8") as file:
        env_args = yaml.load(file, Loader=yaml.FullLoader)
    return algo_args, env_args


def update_args(unparsed_dict, *args):
    """Update loaded config with unparsed command-line arguments.
    Args:
        unparsed_dict: (dict) Unparsed command-line arguments.
        *args: (list[dict]) argument dicts to be updated.
    """
    target_names = ["algo_args", "env_args", "main_args"]
    targets = []
    for idx, args_dict in enumerate(args):
        name = target_names[idx] if idx < len(target_names) else f"args_{idx}"
        targets.append((name, args_dict))

    def _walk_leaf_paths(cfg, prefix=()):
        if not isinstance(cfg, dict):
            return
        for key, value in cfg.items():
            path = prefix + (key,)
            if isinstance(value, dict):
                yield from _walk_leaf_paths(value, path)
            else:
                yield path

    def _path_exists(cfg, path):
        cursor = cfg
        for key in path[:-1]:
            if not isinstance(cursor, dict) or key not in cursor:
                return False
            cursor = cursor[key]
        return isinstance(cursor, dict) and path[-1] in cursor

    def _set_path(cfg, path, value):
        cursor = cfg
        for key in path[:-1]:
            cursor = cursor[key]
        cursor[path[-1]] = value

    def _resolve_key(raw_key):
        selected = targets
        key = raw_key
        explicit = False
        for name, cfg in targets:
            prefix = f"{name}."
            if raw_key.startswith(prefix):
                selected = [(name, cfg)]
                key = raw_key[len(prefix):]
                explicit = True
                break

        if not key:
            return explicit, []

        matches = []
        if "." in key:
            path = tuple(part for part in key.split(".") if part)
            if not path:
                return explicit, []
            for name, cfg in selected:
                if _path_exists(cfg, path):
                    matches.append((name, path))
        else:
            for name, cfg in selected:
                for path in _walk_leaf_paths(cfg):
                    if path[-1] == key:
                        matches.append((name, path))
        return explicit, matches

    unknown_keys = []
    ambiguous_keys = {}

    for raw_key, value in unparsed_dict.items():
        explicit, matches = _resolve_key(raw_key)
        if not matches:
            unknown_keys.append(raw_key)
            continue
        if not explicit and len(matches) > 1:
            ambiguous_keys[raw_key] = [
                f"{name}.{'.'.join(path)}" for name, path in matches
            ]
            continue
        for name, path in matches:
            for target_name, target_cfg in targets:
                if target_name == name:
                    _set_path(target_cfg, path, value)
                    break

    if unknown_keys or ambiguous_keys:
        errors = []
        if unknown_keys:
            errors.append(
                "Unknown override keys: " + ", ".join(sorted(unknown_keys))
            )
        if ambiguous_keys:
            for key in sorted(ambiguous_keys):
                options = ", ".join(sorted(ambiguous_keys[key]))
                errors.append(
                    f"Ambiguous override '{key}', use a fully-qualified path. "
                    f"Candidates: {options}"
                )
        raise ValueError("Invalid config overrides.\n" + "\n".join(errors))


def get_task_name(env, env_args):
    """Get task name."""
    if env == "smac":
        task = env_args["map_name"]
    elif env == "smacv2":
        task = env_args["map_name"]
    elif env == "mamujoco":
        task = f"{env_args['scenario']}-{env_args['agent_conf']}"
    elif env == "pettingzoo_mpe":
        if env_args["continuous_actions"]:
            task = f"{env_args['scenario']}-continuous"
        else:
            task = f"{env_args['scenario']}-discrete"
    elif env == "gym":
        task = env_args["scenario"]
    return task


def init_dir(env, env_args, algo, exp_name, seed, hms_time, logger_path):
    """Init directory for saving results."""
    task = get_task_name(env, env_args)
    exp_path = [
        env,
        task,
        algo,
        f"{exp_name}_seed{seed}_time{hms_time}"
    ]

    run_path = os.path.join(logger_path, "slurm", *exp_path)
    models_path = os.path.join(logger_path, "models", *exp_path)
    os.makedirs(run_path, exist_ok=True)
    os.makedirs(models_path, exist_ok=True)
    return run_path, models_path


def is_json_serializable(value):
    """Check if v is JSON serializable."""
    try:
        json.dumps(value)
        return True
    except Error:
        return False


def convert_json(obj):
    """Convert obj to a version which can be serialized with JSON."""
    if is_json_serializable(obj):
        return obj
    else:
        if isinstance(obj, dict):
            return {convert_json(k): convert_json(v) for k, v in obj.items()}

        elif isinstance(obj, tuple):
            return (convert_json(x) for x in obj)

        elif isinstance(obj, list):
            return [convert_json(x) for x in obj]

        elif hasattr(obj, "__name__") and not ("lambda" in obj.__name__):
            return convert_json(obj.__name__)

        elif hasattr(obj, "__dict__") and obj.__dict__:
            obj_dict = {
                convert_json(k): convert_json(v) for k, v in obj.__dict__.items()
            }
            return {str(obj): obj_dict}

        return str(obj)


def save_config(args, algo_args, env_args, run_dir):
    """Save the configuration of the program."""
    config = {"main_args": args, "algo_args": algo_args, "env_args": env_args}
    config_json = convert_json(config)
    output = json.dumps(config_json, separators=(",", ": "), indent=4, sort_keys=True)
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as out:
        out.write(output)
    return config_json


def args_sanity_check(config, main_args, console_logger, env_args=None):
    # set CUDA flags. Use cuda whenever possible!
    if config["device"]["cuda"] and not th.cuda.is_available():
        config["device"]["cuda"] = False
        console_logger.warning(
            "CUDA flag cuda was switched OFF automatically because no CUDA devices are available!"
        )

    # DISSCv2-aligned defaults for gym LBF/VMAS unless explicitly set.
    if main_args.get("env") == "gym" and env_args is not None:
        scenario = env_args.get("scenario", "")
        if scenario.startswith("lbforaging:") or scenario.startswith("vmas-"):
            if config["eval"].get("eval_episodes", 20) == 20:
                config["eval"]["eval_episodes"] = 100
            if "log_interval_steps" not in config["train"] or config["train"]["log_interval_steps"] is None:
                config["train"]["log_interval_steps"] = 50000
            if "eval_interval_steps" not in config["train"] or config["train"]["eval_interval_steps"] is None:
                config["train"]["eval_interval_steps"] = 50000

    # set eval_episodes to be a multiple of n_eval_rollout_threads
    if config["eval"]["eval_episodes"] < config["eval"]["n_eval_rollout_threads"]:
        config["eval"]["eval_episodes"] = config["eval"]["n_eval_rollout_threads"]
    else:
        config["eval"]["eval_episodes"] = (
            config["eval"]["eval_episodes"] // config["eval"]["n_eval_rollout_threads"]
        ) * config["eval"]["n_eval_rollout_threads"]

    # set eval_interval to be a multiple of log_interval
    if config["train"]["eval_interval"] < config["train"]["log_interval"]:
        config["train"]["eval_interval"] = config["train"]["log_interval"]
    else:
        config["train"]["eval_interval"] = (
            config["train"]["eval_interval"] // config["train"]["log_interval"]
        ) * config["train"]["log_interval"]

    # step-based intervals (when set) take precedence in runners and are aligned here.
    if config["train"].get("log_interval_steps", None) is not None:
        config["train"]["log_interval_steps"] = int(config["train"]["log_interval_steps"])
        if config["train"]["log_interval_steps"] <= 0:
            raise ValueError("train.log_interval_steps must be positive.")
    if config["train"].get("eval_interval_steps", None) is not None:
        config["train"]["eval_interval_steps"] = int(config["train"]["eval_interval_steps"])
        if config["train"]["eval_interval_steps"] <= 0:
            raise ValueError("train.eval_interval_steps must be positive.")
    if (
        config["train"].get("log_interval_steps", None) is not None
        and config["train"].get("eval_interval_steps", None) is not None
    ):
        if config["train"]["eval_interval_steps"] < config["train"]["log_interval_steps"]:
            config["train"]["eval_interval_steps"] = config["train"]["log_interval_steps"]
        else:
            config["train"]["eval_interval_steps"] = (
                config["train"]["eval_interval_steps"]
                // config["train"]["log_interval_steps"]
            ) * config["train"]["log_interval_steps"]

    # mappo_t uses a transformer critic with optional recurrent state, which requires
    # embedding width and recurrent hidden width to match.
    if main_args.get("algo") == "mappo_t":
        model_cfg = config.get("model", {})
        use_recurrent = bool(
            model_cfg.get("use_recurrent_policy", False)
            or model_cfg.get("use_naive_recurrent_policy", False)
        )
        if use_recurrent:
            hidden_sizes = model_cfg.get("hidden_sizes", None)
            transformer_cfg = model_cfg.get("transformer", {})
            if not hidden_sizes:
                raise ValueError(
                    "mappo_t requires model.hidden_sizes to be set when recurrent policy is enabled."
                )
            if "n_embd" not in transformer_cfg:
                raise ValueError(
                    "mappo_t requires model.transformer.n_embd when recurrent policy is enabled."
                )
            rnn_hidden_size = int(hidden_sizes[-1])
            n_embd = int(transformer_cfg["n_embd"])
            if n_embd != rnn_hidden_size:
                raise ValueError(
                    "Invalid mappo_t config: model.transformer.n_embd "
                    f"({n_embd}) must match model.hidden_sizes[-1] ({rnn_hidden_size}) "
                    "when recurrent policy is enabled."
                )

    # set save_interval to be 10*eval_interval if not specified
    if config["train"].get("save_interval", 0) == 0:
        config["train"]["save_interval"] = 10 * config["train"]["eval_interval"]

    # set current time if not specified
    if not main_args["hms_time"]:
        main_args["hms_time"] = datetime.datetime.now().strftime(
            "%Y-%m-%d-%H-%M-%S-%f")
