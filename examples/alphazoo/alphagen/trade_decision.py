from math import isnan
import sys
sys.path.append('examples/alphagen')

import pandas as pd
from alphagen.trade.base import StockPosition, StockStatus
from alphagen_qlib.calculator import QLibStockDataCalculator

from alphagen_qlib.strategy import TopKSwapNStrategy
from alphagen_qlib.utils import load_alpha_pool_by_path, load_recent_data

from qlib.utils import init_instance_by_config
from qlib.contrib.evaluate import (
    backtest_daily as normal_backtest,
    risk_analysis,
    indicator_analysis
)



POOL_PATH = 'examples/alphagen/result/checkpoints/new_all_100_42_20240508125917/100352_steps_pool.json'

def load_data():
    """
    A function to load data using a dataset configuration and return the loaded data along with the end date.
    """
    dataset_conf = {
        "class": "StockData",
        "module_path": "examples.alphagen.alphagen_qlib.stock_data",
        "kwargs": {
            "instrument": "csi300",
            "freq": "day",
            "provider_uri": "D:\\qlib_data\\qlib_cn",
        }
    }

    dataset_test_conf = dataset_conf.copy()
    dataset_test_conf['kwargs']['start_time'] = '2021-01-01'
    dataset_test_conf['kwargs']['end_time'] = '2023-12-31'
    data = init_instance_by_config(dataset_test_conf)

    return data, dataset_test_conf['kwargs']['end_time']

def main():

    dataset_conf = {
        "class": "StockData",
        "module_path": "examples.alphagen.alphagen_qlib.stock_data",
        "kwargs": {
            "instrument": "all",
            "freq": "day",
            "provider_uri": "D:\\qlib_data\\qlib_cn",
        }
    }

    dataset_test_conf = dataset_conf.copy()
    dataset_test_conf['kwargs']['start_time'] = '2021-01-01'
    dataset_test_conf['kwargs']['end_time'] = '2023-12-29'

    data = init_instance_by_config(dataset_test_conf)

    latest_date = dataset_test_conf['kwargs']['end_time']

    print(f'Latest date: {latest_date}')
    calculator = QLibStockDataCalculator(data=data, target=None)
    exprs, weights = load_alpha_pool_by_path(POOL_PATH)

    ensemble_alpha = calculator.make_ensemble_alpha(exprs, weights)
    df = data.make_dataframe(ensemble_alpha)
    df.rename(columns={0: 'score'}, inplace=True)

    strategy_config = {
        "class": "TopkDropoutStrategy",
        "module_path": "qlib.contrib.strategy.signal_strategy",
        "kwargs": {
            "signal": df,
            "topk": 100,
            "n_drop": 10,
        },
    }

    exchange_kwargs = {
        "deal_price": 'close',  # deal_price is None, using C.deal_price
        "open_cost": 0.0005,
        "close_cost": 0.0015,
        "min_cost": 5,
    }



    report_df, positions = normal_backtest(start_time="2021-01-01", end_time='2023-12-31', strategy=strategy_config,
                                           benchmark='SH000985', exchange_kwargs=exchange_kwargs)

    analysis = dict()
    analysis["excess_return_without_cost"] = risk_analysis(report_df["return"] - report_df["bench"])
    analysis["excess_return_with_cost"] = risk_analysis(
        report_df["return"] - report_df["bench"] - report_df["cost"]
    )
    analysis["abs_return_with_cost"] = risk_analysis(report_df["return"] - report_df["cost"])

# if __name__ == '__main__':
#     main()