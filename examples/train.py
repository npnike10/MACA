"""Train an algorithm."""
import argparse
import json
import os
from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver, MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds
from harl.common.base_logger import get_logger
from harl.utils.configs_tools import (
    get_defaults_yaml_args,
    update_args,
    get_task_name,
    args_sanity_check,
)
from harl.runners import RUNNER_REGISTRY

from warnings import filterwarnings
filterwarnings("ignore")

SETTINGS["CAPTURE_MODE"] = "fd" # set to "no" if you want to see stdout/stderr in console
SETTINGS.CONFIG.READ_ONLY_CONFIG = False
logger = get_logger()
ex = Experiment("harl_xplr")
ex.logger = logger
ex.captured_out_filter = apply_backspaces_and_linefeeds


@ex.main
def my_main(_run, _config, _log):
    args, algo_args, env_args = (
        _config["main_args"],
        _config["algo_args"],
        _config["env_args"],
    )

    runner = None
    exit_code = 0
    try:
        # start training
        runner = RUNNER_REGISTRY[args["algo"]](
            args, algo_args, env_args, _run, _log,
        )
        runner.run()
    except Exception:
        exit_code = 1
        raise
    finally:
        if runner is not None:
            runner.close(exit_code=exit_code)

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="happo",
        choices=[
            "happo",
            "mappo",
            "mappo_t",
            "ippo",
            "coma",
            "mappo_vd",
        ],
    )
    parser.add_argument(
        "--env",
        type=str,
        default="pettingzoo_mpe",
        choices=[
            "smac",
            "mamujoco",
            "pettingzoo_mpe",
            "gym",
            "smacv2",
        ],
    )
    parser.add_argument(
        "--exp_name", type=str, default="installtest", help="Experiment name."
    )
    parser.add_argument(
        "--load_config",
        type=str,
        default="",
        help="If set, load existing experiment config file instead of reading from yaml config file.",
    )
    parser.add_argument(
        "--hms_time",
        type=str,
        default="",
        help="Use this time along with the experiment name to uniquely indentify one experiment. Its role is similar to a hyperparameter token.",
    )
    args, unparsed_args = parser.parse_known_args()

    def process(arg):
        try:
            return eval(arg)
        except Exception:
            return arg

    def parse_unparsed_args(raw_args):
        parsed = {}
        idx = 0
        while idx < len(raw_args):
            token = raw_args[idx]
            if "=" in token:
                key, raw_value = token.split("=", 1)
                key = key.lstrip("-")
                value = process(raw_value)
                idx += 1
            else:
                if not token.startswith("-"):
                    raise ValueError(
                        f"Invalid override token '{token}'. "
                        "Use '--key=value' or '--key value'."
                    )
                if idx + 1 >= len(raw_args):
                    raise ValueError(f"Missing value for override '{token}'.")
                key = token.lstrip("-")
                value = process(raw_args[idx + 1])
                idx += 2

            if key in parsed:
                raise ValueError(f"Duplicate override key '{key}'.")
            parsed[key] = value
        return parsed

    unparsed_dict = parse_unparsed_args(unparsed_args)
    args = vars(args)  # convert to dict
    if args["load_config"] != "":  # load config from existing config file
        with open(args["load_config"], encoding="utf-8") as file:
            all_config = json.load(file)
        algo_args = all_config["algo_args"]
        env_args = all_config["env_args"]
    else:  # load config from corresponding yaml file
        algo_args, env_args = get_defaults_yaml_args(args["algo"], args["env"])
    update_args(unparsed_dict, algo_args, env_args)  # update args from command line

    args_sanity_check(algo_args, args, logger, env_args=env_args)
    all_config = {
        "main_args": args,
        "algo_args": algo_args,
        "env_args": env_args,
    }
    # now add all the config to sacred
    ex.add_config(all_config)

    # Save to disk by default for sacred
    logger.info("Saving to FileStorageObserver in results/sacred.")
    task_name = get_task_name(args["env"], env_args)
    sacred_logs_dir = os.path.join(
        algo_args["logger"]["log_dir"], "sacred",
        args["env"], task_name, args["algo"],
        f'{args["exp_name"]}_seed{algo_args["seed"]["seed"]}_time{args["hms_time"]}'
    )

    ex.observers.append(FileStorageObserver.create(sacred_logs_dir))
    ex.run_commandline("dummy")


if __name__ == "__main__":
    main()
