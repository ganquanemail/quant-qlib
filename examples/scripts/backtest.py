import os
import pickle as pkl
import pandas as pd
import numpy as np
import pathlib
import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config, flatten_dict
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord
from qlib.tests.data import GetData
from qlib.tests.config import CSI300_BENCH, CSI300_GBDT_TASK
from qlib.tests.config import get_dataset_config


from qlib.contrib.eva.alpha import calc_ic, calc_long_short_return

from qlib.contrib.report import analysis_model, analysis_position
from qlib.data import D

from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy
from qlib.contrib.evaluate import (
    backtest_daily as normal_backtest,
    risk_analysis,
    indicator_analysis
)

import plotly.io as pio

provider_uri = "D:\qlib_data\qlib_cn"  # target_dir

def prepare_input():
    ### 生成模拟信号
    GetData().qlib_data(target_dir=provider_uri, region=REG_CN, exists_skip=True)
    qlib.init(provider_uri=provider_uri, region=REG_CN)

    # model = init_instance_by_config(CSI300_GBDT_TASK["model"])
    dataset = init_instance_by_config(CSI300_GBDT_TASK["dataset"])

    example_df = dataset.prepare("test")

    prediction = example_df.loc[:, ['LABEL0', 'KMID']]
    prediction.columns = ['label', 'score']
    prediction = prediction.swaplevel()

    return prediction


def run():

    # GetData().qlib_data(target_dir=provider_uri, region=REG_CN, exists_skip=True)
    qlib.init(provider_uri=provider_uri, region=REG_CN)

    model = 'DLinear'
    model_id = '93fbbe0aa2bd41baaf6fe1d811561b1e'
    fpath = pathlib.Path("""\\examples\\benchmarks\\{}\\mlruns\\1\\{}\\artifacts\\pred.pkl""".format(model,model_id))


    freq = 5
    topk = 400
    n_drop = 40
    benchmark = 'SH000985'

    pred = pd.read_pickle(fpath)
    pred = pred.unstack().iloc[::freq, :].stack()
    pred = pred.loc[:,['score']]

    strategy_config = {
        "class": "TopkDropoutStrategy",
        "module_path": "qlib.contrib.strategy.signal_strategy",
        "kwargs": {
            "signal": pred,
            "topk": topk,
            "n_drop": n_drop,
        },
    }

    exchange_kwargs = {
        "deal_price": 'close',  # deal_price is None, using C.deal_price
        "open_cost": 0.0005,
        "close_cost": 0.0015,
        "min_cost": 5,
    }


    report_df, positions = normal_backtest(start_time="2021-01-01", end_time="2023-03-01", strategy=strategy_config,
                                           benchmark=benchmark, exchange_kwargs=exchange_kwargs)

    ## 生成净值HTML
    fig = analysis_position.report_graph(report_df, show_notebook=False)
    fig[0].write_html(
        'backtest_result_{}_single_d{}_top{}_drop{}_alpha158.html'.format( model, freq, topk, n_drop))

    ## 计算指标
    analysis = dict()
    analysis["excess_return_without_cost"] = risk_analysis(report_df["return"] - report_df["bench"])
    analysis["excess_return_with_cost"] = risk_analysis(
        report_df["return"] - report_df["bench"] - report_df["cost"]
    )
    analysis["abs_return_with_cost"] = risk_analysis(report_df["return"] - report_df["cost"])

    print(analysis)
