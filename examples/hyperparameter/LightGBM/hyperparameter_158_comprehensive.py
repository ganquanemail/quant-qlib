import sys
import os
import socket
import logging

from pathlib import Path

import qlib
import optuna
from optuna.trial import TrialState
from functools import partial

import fire
import ruamel.yaml as yaml
from qlib.config import C
from qlib.model.trainer import task_train
from qlib.utils.data import update_config
from qlib.log import get_module_logger
from qlib.utils import set_log_with_config
from qlib.utils import init_instance_by_config
from qlib.utils import auto_filter_kwargs
from qlib.utils import fill_placeholder
from qlib.utils import flatten_dict

from qlib.model.base import Model

from qlib.workflow import R
from qlib.workflow.recorder import Recorder
from qlib.workflow.task.manage import TaskManager, run_task

set_log_with_config(C.logging_config)
logger = get_module_logger("qrun", logging.INFO)

def get_path_list(path):
    if isinstance(path, str):
        return [path]
    else:
        return list(path)

def _log_task_info(task_config: dict):
    R.log_params(**flatten_dict(task_config))
    R.save_objects(**{"task": task_config})  # keep the original format and datatype
    R.set_tags(**{"hostname": socket.gethostname()})

def sys_config(config, config_path):
    """
    Configure the `sys` section

    Parameters
    ----------
    config : dict
        configuration of the workflow.
    config_path : str
        path of the configuration
    """
    sys_config = config.get("sys", {})

    # abspath
    for p in get_path_list(sys_config.get("path", [])):
        sys.path.append(p)

    # relative path to config path
    for p in get_path_list(sys_config.get("rel_path", [])):
        sys.path.append(str(Path(config_path).parent.resolve().absolute() / p))

def _log_task_info(task_config: dict):
    R.log_params(**flatten_dict(task_config))
    R.save_objects(**{"task": task_config})  # keep the original format and datatype
    R.set_tags(**{"hostname": socket.gethostname()})


def workflow(config_path, experiment_name="workflow", uri_folder="mlruns"):
    """
    This is a Qlib CLI entrance.
    User can run the whole Quant research workflow defined by a configure file
    - the code is located here ``qlib/workflow/cli.py`

    User can specify a base_config file in your workflow.yml file by adding "BASE_CONFIG_PATH".
    Qlib will load the configuration in BASE_CONFIG_PATH first, and the user only needs to update the custom fields
    in their own workflow.yml file.

    For examples:

        qlib_init:
            provider_uri: "~/.qlib/qlib_data/cn_data"
            region: cn
        BASE_CONFIG_PATH: "workflow_config_lightgbm_Alpha158_csi500.yaml"
        market: csi300

    """
    with open(config_path) as fp:
        config = yaml.safe_load(fp)

    base_config_path = config.get("BASE_CONFIG_PATH", None)
    if base_config_path:
        logger.info(f"Use BASE_CONFIG_PATH: {base_config_path}")
        base_config_path = Path(base_config_path)

        # it will find config file in absolute path and relative path
        if base_config_path.exists():
            path = base_config_path
        else:
            logger.info(
                f"Can't find BASE_CONFIG_PATH base on: {Path.cwd()}, "
                f"try using relative path to config path: {Path(config_path).absolute()}"
            )
            relative_path = Path(config_path).absolute().parent.joinpath(base_config_path)
            if relative_path.exists():
                path = relative_path
            else:
                raise FileNotFoundError(f"Can't find the BASE_CONFIG file: {base_config_path}")

        with open(path) as fp:
            base_config = yaml.safe_load(fp)
        logger.info(f"Load BASE_CONFIG_PATH succeed: {path.resolve()}")
        config = update_config(base_config, config)

    # config the `sys` section
    sys_config(config, config_path)

    if "exp_manager" in config.get("qlib_init"):
        qlib.init(**config.get("qlib_init"))
    else:
        exp_manager = C["exp_manager"]
        exp_manager["kwargs"]["uri"] = "file:" + str(Path(os.getcwd()).resolve() / uri_folder)
        qlib.init(**config.get("qlib_init"), exp_manager=exp_manager)

    if "experiment_name" in config:
        experiment_name = config["experiment_name"]
    recorder = task_train(config.get("task"), experiment_name=experiment_name)

    recorder.save_objects(config=config)
def objective(trial, dataset, experiment_name, config):
    task = {
        "model": {
            "class": "LGBModel",
            "module_path": "qlib.contrib.model.gbdt",
            "kwargs": {
                "loss": "mse",
                "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.5, 1),
                "learning_rate": trial.suggest_uniform("learning_rate", 0, 1),
                "subsample": trial.suggest_uniform("subsample", 0, 1),
                "lambda_l1": trial.suggest_loguniform("lambda_l1", 1e-8, 1e4),
                "lambda_l2": trial.suggest_loguniform("lambda_l2", 1e-8, 1e4),
                "max_depth": 10,
                "num_boost_round": 100,
                "num_leaves": trial.suggest_int("num_leaves", 1, 1024),
                "feature_fraction": trial.suggest_uniform("feature_fraction", 0.4, 1.0),
                "bagging_fraction": trial.suggest_uniform("bagging_fraction", 0.4, 1.0),
                "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 50),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            },
        },
    }

    with R.start(experiment_name=None, recorder_name=None):
        _log_task_info(task)
        rec = R.get_recorder()
        logger.info(f"recorder info : {rec}")
        evals_result = dict()
        model: Model = init_instance_by_config(task["model"], accept_types=Model)
        model.fit(dataset, evals_result=evals_result)
        R.save_objects(**{"params.pkl": model})
        placehorder_value = {"<MODEL>": model, "<DATASET>": dataset}
        task_config = fill_placeholder(config.get("task"), placehorder_value)
        records = task_config.get("record", [])
        if isinstance(records, dict):  # prevent only one dict
            records = [records]
        for record in records:
            # Some recorder require the parameter `model` and `dataset`.
            # try to automatically pass in them to the initialization function
            # to make defining the tasking easier
            r = init_instance_by_config(
                record,
                recorder=rec,
                default_module="qlib.workflow.record_temp",
                try_kwargs={"model": model, "dataset": dataset},
            )
            r.generate()

        return min(evals_result["valid"]['l2'])


def run():
    experiment_name = 'LGBM_158'
    uri_folder = "mlruns"
    config_path = 'examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml'
    try:
        with open(config_path) as fp:
            config = yaml.safe_load(fp)
    except:
        CUR_DIR = Path('.').resolve()
        PAR_DIR = CUR_DIR.parent.parent.parent
        config_path = os.path.join(PAR_DIR, config_path)
        with open(config_path) as fp:
            config = yaml.safe_load(fp)

    base_config_path = config.get("BASE_CONFIG_PATH", None)
    if base_config_path:
        logger.info(f"Use BASE_CONFIG_PATH: {base_config_path}")
        base_config_path = Path(base_config_path)

        # it will find config file in absolute path and relative path
        if base_config_path.exists():
            path = base_config_path
        else:
            logger.info(
                f"Can't find BASE_CONFIG_PATH base on: {Path.cwd()}, "
                f"try using relative path to config path: {Path(config_path).absolute()}"
            )
            relative_path = Path(config_path).absolute().parent.joinpath(base_config_path)
            if relative_path.exists():
                path = relative_path
            else:
                raise FileNotFoundError(f"Can't find the BASE_CONFIG file: {base_config_path}")

        with open(path) as fp:
            base_config = yaml.safe_load(fp)
        logger.info(f"Load BASE_CONFIG_PATH succeed: {path.resolve()}")
        config = update_config(base_config, config)

    # config the `sys` section
    sys_config(config, config_path)

    if "exp_manager" in config.get("qlib_init"):
        qlib.init(**config.get("qlib_init"))
    else:
        exp_manager = C["exp_manager"]
        exp_manager["kwargs"]["uri"] = "file:" + str(Path(os.getcwd()).resolve() / uri_folder)
        qlib.init(**config.get("qlib_init"), exp_manager=exp_manager)

    study = optuna.Study(study_name="LGBM_158", storage="sqlite:///db.sqlite3")

    dataset = init_instance_by_config(config['task']['dataset'])

    objective_partial = partial(objective, dataset=dataset, experiment_name=experiment_name, config=config)
    study.optimize(objective_partial, n_jobs=1, n_trials=1)

    # pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    # complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    # logger.info("Study statistics: ")
    # logger.info("  Number of finished trials: ", len(study.trials))
    # logger.info("  Number of pruned trials: ", len(pruned_trials))
    # logger.info("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    logger.info("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

if __name__ == "__main__":
    run()