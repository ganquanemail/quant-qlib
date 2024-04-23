import abc
import platform
import pyodbc
import pymongo

import pandas as pd
import numpy as np
from datetime import datetime
from dateutil import parser

# your database config
config = {}


def getMongoClient(host = 'localhost', port = 27017):
    conn = pymongo.MongoClient(host=host,
                               port=port)
    return conn

class BaseVendor(object):
    __class__ = abc.ABCMeta

    @abc.abstractmethod
    def getStockBasic(self, idate):
        raise NotImplementedError

    @abc.abstractmethod
    def getBar(self, ticker, frequency, fromDate, Todate, useDividendAdjust):
        raise NotImplementedError

    @abc.abstractmethod
    def getTick(self, ticker, fromDate, toDate):
        raise NotImplementedError

    @abc.abstractmethod
    def getDividend(self, ticker, start_date, end_date):
        raise NotImplementedError

    @abc.abstractmethod
    def getBalanceSheet(self, ticker, date, rpt_type):
        raise NotImplementedError

    @abc.abstractmethod
    def getIncomeStatement(self, ticker, rpt_date, rpt_type):
        raise NotImplementedError

    @abc.abstractmethod
    def getCashFlow(self, ticker, rpt_date, rpt_type):
        raise NotImplementedError

    @abc.abstractmethod
    def updateBalanceSheet(self):
        raise NotImplementedError

    @abc.abstractmethod
    def updateIncomeStatement(self):
        raise NotImplementedError

    @abc.abstractmethod
    def updateCashFlowStatement(self):
        raise NotImplementedError

def test_conn():
    conn = pyodbc.connect(
        r'DRIVER={0};SERVER={1};DATABASE={2};UID={3};PWD={4}'.format(config['driver'], config['server'],
                                                                     config['database'], config['uid'],
                                                                     config['password']))
    return conn


def code2ticker(code):
    if int(code[0])>3:
        return code + '.SH'
    elif int(code[0])<4:
        return code + '.SZ'

class sqlserverVendor(BaseVendor):
    def __init__(self, config=config, qry=None, database=None):
        self.config = config
        self.__qry = qry
        if database is not None:
            self.config['database'] = database
        pass

    @property
    def qry(self):
        return self.__qry

    @qry.setter
    def qry(self, qry):
        self.__qry = qry

    def Build_DB_Connnection(self):
        ct_server = self.config['server']
        ct_database  = self.config['database']
        ct_uid    = self.config['uid']
        ct_passwd  = self.config['password']
        ct_driver = self.config['driver']
        conn = pyodbc.connect(r'DRIVER={0};SERVER={1};DATABASE={2};UID={3};PWD={4}'.format(ct_driver, ct_server, ct_database, ct_uid, ct_passwd ))
        return conn

    def getDBdata(self, column_name, table_name, startDate, endDate, condition=None, dt_label='trade_dt'):

        assert (isinstance(column_name, list))

        column_name = [icol.upper() for icol in column_name]
        table_name  = table_name.upper()
        fields = ','.join(column_name)
        dt_label = dt_label.upper()
        sql = "select {} from {} ".format(fields, table_name)
        if condition is None:
            sql += 'where OBJECT_ID is not null'
        else:
            assert (isinstance(condition, str))
            sql += condition
        if startDate is not None:
            sql += ' and {} >= \'\'{}\'\''.format(dt_label, startDate)
        if endDate is not None:
            sql += ' and {} <= \'\'{}\'\''.format(dt_label, endDate)
        else:
            endDate = datetime.today().date()
            endDate = str(endDate)
            endDate = ''.join(endDate.split('-'))
            sql += ' and {} <= \'\'{}\'\''.format(dt_label, endDate)

        db = self.Build_DB_Connnection()
        cursor = db.cursor()
        cursor.execute(sql)
        data = cursor.fetchall()
        df = pd.DataFrame(np.array(data))
        column_name = [icol.lower() for icol in column_name]
        df.columns = column_name

        return df

    def getDBtable_columns(self, table_name):

        db = self.Build_DB_Connnection()
        cursor = db.cursor()
        sql = "select top 10 * from wande.dbo.{}".format(table_name)
        cursor.execute(sql)
        column_names = cursor.description
        column_list = []
        for i in range(len(column_names)):
            column_list.append(column_names[i][0])
        column_list = [icol.lower() for icol in column_list]
        return column_list

    def getBar(self, ticker, frequency, fromDate, toDate, useDividendAdjust=None, isIndex=False):
        if frequency != 60 * 60 * 24:
            raise Exception('only support day frequency')

        if isinstance(ticker, str):
            if ticker.endswith('WI') or ticker in ['000001.SH', '000905.SH', '000906.SH', '000985.SH', '000852.SH', '399303.SZ'
                                '000300.SH', '000016.SH','000852.SH', '399102.SZ']:
                isIndex = True

        if not isIndex:
            if isinstance(ticker, str):
                column_name = ['s_info_windcode', 'trade_dt', 's_dq_preclose', 's_dq_open', 's_dq_close', 's_dq_high', 's_dq_low', 's_dq_pctchange', 's_dq_volume', 's_dq_amount', 's_dq_adjfactor', 's_dq_avgprice', 's_dq_tradestatuscode', 's_dq_limit', 's_dq_stopping']
                bar = self.getDBdata(column_name, 'ashareeodprices', fromDate, toDate, 'where S_INFO_WINDCODE=\'{}\''.format(ticker), dt_label='trade_dt')
                bar.columns = ['ticker', 'datetime', 'preclose', 'open', 'close', 'high', 'low', 'pctchange', 'volume', 'amount', 'adjfactor', 'avgprice', 'tradestatuscode', 'limit', 'stopping']

                bar['datetime'] = pd.to_datetime(bar.datetime)
                bar = bar.sort_values(by='datetime')
                bar.loc[:,
                ['open', 'close', 'high', 'low',  'volume', 'amount']] =  bar.loc[:,
                ['open', 'close', 'high', 'low',  'volume', 'amount']].astype(float)
                bar.amount = bar.amount * 1000
                bar.index = list(range(len(bar)))
            else:
                column_name = ['s_info_windcode', 'trade_dt', 's_dq_preclose', 's_dq_open',
                               's_dq_close', 's_dq_high', 's_dq_low', 's_dq_pctchange', 's_dq_volume', 's_dq_amount',
                               's_dq_adjfactor', 's_dq_avgprice', 's_dq_tradestatuscode', 's_dq_limit', 's_dq_stopping']
                bar = self.getDBdata(column_name, 'ashareeodprices', fromDate, toDate, dt_label='trade_dt')
                bar.columns = ['ticker', 'datetime', 'preclose', 'open', 'close', 'high', 'low',
                               'pctchange', 'volume', 'amount', 'adjfactor', 'avgprice', 'tradestatuscode', 'limit',
                               'stopping']
                bar['datetime'] = pd.to_datetime(bar.datetime)
                bar = bar.sort_values(by='datetime')
                bar.loc[:,
                ['open', 'close', 'high', 'low',  'volume', 'amount']] =  bar.loc[:,
                ['open', 'close', 'high', 'low',  'volume', 'amount']].astype(float)
                bar.amount = bar.amount * 1000

                # select columns
                bar = bar.loc[:,['ticker', 'datetime', 'open', 'close', 'high', 'low',  'volume', 'amount']]
        else:
            if ticker.endswith('WI'):
                table_name = 'AINDEXWINDINDUSTRIESEOD'
            else:
                table_name = 'AIndexEODPrices'
            column_name = ['s_info_windcode', 'trade_dt', 's_dq_open', 's_dq_close', 's_dq_high', 's_dq_low', 's_dq_volume', 's_dq_amount']
            bar = self.getDBdata(column_name, table_name, fromDate, toDate, 'where S_INFO_WINDCODE= \'\'{}\'\''.format(ticker), dt_label='trade_dt')
            bar.columns = ['ticker', 'datetime', 'open', 'close', 'high', 'low',  'volume', 'amount']
            bar['datetime'] = pd.to_datetime(bar.datetime)
            bar = bar.sort_values(by='datetime')
            bar.loc[:,['open', 'close', 'high', 'low',  'volume', 'amount']] =  bar.loc[:, ['open', 'close', 'high', 'low',  'volume', 'amount']].astype(float)
            bar.amount = bar.amount * 1000
            bar.index = list(range(len(bar)))
        return bar

    def gettdays(self, startDate=None, endDate=None, exchmarket='SSE'):
        sql = 'select trade_days from asharecalendar where s_info_exchmarket=\'{}\''.format(exchmarket)
        if startDate is not None:
            sql += ' and trade_days >= \'{}\''.format(startDate)
        if endDate is not None:
            sql += ' and trade_days <= \'{}\''.format(endDate)
        else:
            endDate = datetime.today().date()
            endDate = str(endDate)
            endDate = ''.join(endDate.split('-'))
            sql += ' and trade_days <= \'{}\''.format(endDate)

        conn = self.Build_DB_Connnection()
        cursor = conn.cursor()
        cursor.execute(sql)
        res = cursor.fetchall()
        res = np.array(res).flatten().tolist()
        res.sort()

        return res

    def getIndexCompos(self, ticker, db_name='AINDEXMEMBERS', all_ticker=False, ticker_name='S_INFO_WINDCODE', **kwargs):

        db_name = db_name.upper()

        if ticker is None:
            pass
        elif ticker.endswith('WI'):
            db_name = 'wande.dbo.' +  'aindexmemberswind'.upper()
            ticker_name = 'f_info_windcode'.upper()
        else:
            pass
        sql = "select {0},S_CON_WINDCODE, S_CON_INDATE, S_CON_OUTDATE from {1}".format(ticker_name, db_name)
        if not all_ticker:
            sql += ' where {0} = \'\'{1}\'\''.format(ticker_name, ticker)

        for ikey, ival in kwargs.items():
            if ikey.lower() == 'conn':
                pass
            else:
                if 'where' in sql:
                    condition = ' and'
                else:
                    condition = ' where'
                if isinstance(ival, str):
                    sql += '%s %s = \'\'%s\'\'' % (condition, ikey, ival)
                elif isinstance(ival, list):
                    sql += '%s %s in (' % (condition, ikey)
                    for isec in ival:
                        sql += '\'\'%s\'\',' % isec
                    sql = sql[:-1] + ')'
                else:
                    pass

        conn = self.Build_DB_Connnection()
        cursor = conn.cursor()
        cursor.execute(sql)
        data = cursor.fetchall()
        df = pd.DataFrame(np.array(data), columns=[ticker_name, 's_con_windcode', 's_con_indate', 's_con_outdate'])

        return df

    def getBasicIndicator(self, field, fromDate, toDate):
        column_name = ['s_info_windcode', 'trade_dt', 's_dq_adjfactor']
        data = self.getDBdata(column_name, 'ashareeodprices', fromDate, toDate, dt_label='trade_dt')
        df = pd.DataFrame(list(data), columns=['ticker', 'datetime', 'adj_factor'])
        return df


