# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import datetime
import pandas as pd
from qlib.utils import sqlserver_v
from scripts.dump_bin import DumpDataUpdate

# scv path
PATH = 'F:\\qlib_data\\qlib_csv\\'
# bin path
TARGET_PATH = 'D:\\qlib_data\\qlib_cn\\'

def run():

    wdates = load_calander()
    start = (wdates[-1] + datetime.timedelta(days=1)).strftime('%Y%m%d')

    now = datetime.datetime.now()

    if now.hour >= 16:
        end = now.strftime('%Y%m%d')
    else:
        end = (now - datetime.timedelta(days=1)).strftime('%Y%m%d')

    # stock
    resp = load_data(start, end)
    groupers = resp.groupby('symbol')
    CSV_PATH = os.path.join(PATH, 'qlib_csv_instrument_{}'.format(end))
    # if not exist, create the folder
    if not os.path.exists(CSV_PATH):
        os.makedirs(CSV_PATH)

    for ikey, ival in groupers:
        if ikey.startswith('BJ'):
            pass
        else:
            ival.drop('symbol',axis=1).to_csv(os.path.join(CSV_PATH, ikey+'.csv'), index=None)

    # index
    resp_index = load_index(start, end)

    groupers = resp_index.groupby('symbol')
    for ikey, ival in groupers:
        ival.drop('symbol', axis=1).to_csv(os.path.join(CSV_PATH, ikey + '.csv'), index=None)

    return end

def load_calander():

    CALANEDER_PATH = os.path.join(TARGET_PATH, 'calendars', 'day.txt')
    dates = pd.read_csv(CALANEDER_PATH,header=None)
    dates = pd.to_datetime(dates.values.flatten())

    return dates

def load_data(start, end):

    vendor = sqlserver_v.sqlserverVendor()

    # load quota
    fields = ['S_INFO_WINDCODE', 'TRADE_DT', 'S_DQ_ADJOPEN', 'S_DQ_ADJHIGH', 'S_DQ_ADJLOW', 'S_DQ_ADJCLOSE',
              'S_DQ_ADJFACTOR', 'S_DQ_AVGPRICE', 'S_DQ_VOLUME', 'S_DQ_AMOUNT', 'S_DQ_TRADESTATUSCODE']
    table_name = 'AShareEODPrices'

    _resp = vendor.getDBdata(fields, table_name, start, end)

    _resp.columns = ['symbol', 'date',  'open',  'high', 'low', 'close', 'factor', 'vwap', 'volume', 'total_turnover', 'tradestatus']
    _resp.symbol = _resp.symbol.apply(lambda x : x[-2:] + x[:-3])
    _resp.date = pd.to_datetime(_resp.date)

    _resp.volume = _resp.volume * 100
    _resp.total_turnover = _resp.total_turnover * 1000

    _resp.vwap = _resp.vwap * _resp.factor
    _resp.volume = _resp.volume / _resp.factor

    return _resp


def load_index(start, end):

    vendor = sqlserver_v.sqlserverVendor()

    tickers = ['000016.SH', '000300.SH', '000905.SH', '000906.SH', '000852.SH','399303.SZ', '000985.SH']

    fields = ['s_info_windcode', 'trade_dt', 's_dq_open', 's_dq_close', 's_dq_high', 's_dq_low', 's_dq_volume',
                   's_dq_amount']
    table_name = 'AIndexEODPrices'

    #
    CSV_PATH = os.path.join(PATH, 'qlib_csv_index_{}'.format(end))
    # if not exist, create the folder
    if not os.path.exists(CSV_PATH):
        os.makedirs(CSV_PATH)

    f = lambda x : x[-2:] + x[:-3]

    resps = []

    for ticker in tickers:
        i_resp = vendor.getDBdata(fields, table_name, start, end, condition='where S_INFO_WINDCODE = \'\'{}\'\''.format(ticker))
        i_resp.columns = ['symbol', 'date',  'open', 'close', 'high', 'low', 'volume','total_turnover']

        i_resp.date = pd.to_datetime(i_resp.date)

        i_resp.volume = i_resp.volume * 100
        i_resp.total_turnover = i_resp.total_turnover * 1000

        i_resp['symbol'] = i_resp.symbol.apply(lambda x : f(x))
        # i_resp.drop('symbol', axis=1, inplace=True)
        i_resp = i_resp.sort_values(by='date', ascending=True).dropna()
        i_resp['factor'] = 1

        resps.append(i_resp)

    resp_index = pd.concat(resps, axis=0)

    return  resp_index

def update_instrument(ticker='881001.WI', index_name='all'):
    """
    update instruments
    """
    vendor = sqlserver_v.sqlserverVendor()
    comps = vendor.getIndexCompos(ticker)

    if 'F_INFO_WINDCODE' in comps.columns:
        comps.drop('F_INFO_WINDCODE', axis=1, inplace=True)
    if 'S_INFO_WINDCODE' in comps.columns:
        comps.drop('S_INFO_WINDCODE', axis=1, inplace=True)

    comps['s_con_windcode'] = comps['s_con_windcode'].apply(lambda x : x[-2:] + x[:-3])

    txt_Filepath = 'D:\qlib_data\qlib_cn\instruments\\{}.txt'.format(index_name)

    Note = open(txt_Filepath, mode='w')
    for ikey, row in comps.iterrows():
        if row[0].startswith('BJ'):
            continue
        else:
            if row[2] is None:
                str_data = str(row[0]) + '\t' + pd.to_datetime(row[1]).strftime('%Y-%m-%d')
            else:
                str_data = str(row[0]) + '\t' + pd.to_datetime(row[1]).strftime('%Y-%m-%d') + '\t' + pd.to_datetime(row[2]).strftime('%Y-%m-%d')
            Note.write(str_data)
            Note.write('\n')

if __name__ == '__main__':
    end = run()
    update_instrument()
    csv_path = os.path.join(PATH, 'qlib_csv_instrument_{}'.format(end))
    dp = DumpDataUpdate(csv_path=csv_path, qlib_dir=TARGET_PATH, include_fields=["open","close","high","low","vwap","volume","total_turnover","factor","tradestatus"])
    dp.dump()


