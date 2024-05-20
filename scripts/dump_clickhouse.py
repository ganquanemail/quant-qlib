import re
import abc
import shutil
import numpy as np
import pandas as pd

from tqdm import tqdm
from pathlib import Path
from loguru import logger
from functools import partial
from typing import Iterable, List, Union
from clickhouse_connect import get_client
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor

from qlib.utils import fname_to_code, code_to_fname

db_config = {
    'host' : 'localhost',
    'port': 8133,
}
CLIENT = get_client(**db_config)

def extract_strings(s):
    # pattern = r'\b\d{6}\.(XSHE|XSHG|SH|SZ)\b'
    pattern = r'\b\d{6}\.(?:XSHE|XSHG|SH|SZ)\b'
    matches = re.findall(pattern, s)
    return matches

def format_ticker_to_instrument(ticker):
    if ticker.endswith('.XSHG'):
        return 'SH' + ticker[:-5]
    elif ticker.endswith('.XSHE'):
        return 'SZ' + ticker[:-5]
    elif ticker.endswith('.SH') or ticker.endswith('.SZ'):
        return ticker[-2:] + ticker[:-3]
    else:
        raise ValueError('ticker format error')


class DumpDataBase:
    INSTRUMENTS_START_FIELD = "start_datetime"
    INSTRUMENTS_END_FIELD = "end_datetime"
    CALENDARS_DIR_NAME = "calendars"
    FEATURES_DIR_NAME = "features"
    INSTRUMENTS_DIR_NAME = "instruments"
    DUMP_FILE_SUFFIX = ".bin"
    DAILY_FORMAT = "%Y-%m-%d"
    HIGH_FREQ_FORMAT = "%Y-%m-%d %H:%M:%S"
    INSTRUMENTS_SEP = "\t"
    INSTRUMENTS_FILE_NAME = "all.txt"

    UPDATE_MODE = "update"
    ALL_MODE = "all"

    def __init__(
        self,
        csv_path: str,
        qlib_dir: str,
        backup_dir: str = None,
        freq: str = "day",
        max_workers: int = 32,
        date_field_name: str = "date",
        file_suffix: str = ".csv",
        symbol_field_name: str = "symbol",
        exclude_fields: str = "",
        include_fields: str = "",
        limit_nums: int = None,
    ):
        """

        Parameters
        ----------
        csv_path: str
            stock data path or directory
        qlib_dir: str
            qlib(dump) data director
        backup_dir: str, default None
            if backup_dir is not None, backup qlib_dir to backup_dir
        freq: str, default "day"
            transaction frequency
        max_workers: int, default None
            number of threads
        date_field_name: str, default "date"
            the name of the date field in the csv
        file_suffix: str, default ".csv"
            file suffix
        symbol_field_name: str, default "symbol"
            symbol field name
        include_fields: tuple
            dump fields
        exclude_fields: tuple
            fields not dumped
        limit_nums: int
            Use when debugging, default None
        """
        csv_path = Path(csv_path).expanduser()
        if isinstance(exclude_fields, str):
            exclude_fields = exclude_fields.split(",")
        if isinstance(include_fields, str):
            include_fields = include_fields.split(",")
        self._exclude_fields = tuple(filter(lambda x: len(x) > 0, map(str.strip, exclude_fields)))
        self._include_fields = tuple(filter(lambda x: len(x) > 0, map(str.strip, include_fields)))
        self.file_suffix = file_suffix
        self.symbol_field_name = symbol_field_name
        self.csv_files = sorted(csv_path.glob(f"*{self.file_suffix}") if csv_path.is_dir() else [csv_path])
        if limit_nums is not None:
            self.csv_files = self.csv_files[: int(limit_nums)]
        self.qlib_dir = Path(qlib_dir).expanduser()
        self.backup_dir = backup_dir if backup_dir is None else Path(backup_dir).expanduser()
        if backup_dir is not None:
            self._backup_qlib_dir(Path(backup_dir).expanduser())

        self.freq = freq
        self.calendar_format = self.DAILY_FORMAT if self.freq == "day" else self.HIGH_FREQ_FORMAT

        self.works = max_workers
        self.date_field_name = date_field_name

        self._calendars_dir = self.qlib_dir.joinpath(self.CALENDARS_DIR_NAME)
        self._features_dir = self.qlib_dir.joinpath(self.FEATURES_DIR_NAME)
        self._instruments_dir = self.qlib_dir.joinpath(self.INSTRUMENTS_DIR_NAME)

        self._calendars_list = []

        self._mode = self.ALL_MODE
        self._kwargs = {}

    def _backup_qlib_dir(self, target_dir: Path):
        shutil.copytree(str(self.qlib_dir.resolve()), str(target_dir.resolve()))

    def _format_datetime(self, datetime_d: [str, pd.Timestamp]):
        datetime_d = pd.Timestamp(datetime_d)
        return datetime_d.strftime(self.calendar_format)

    def _get_date(
        self, file_or_df: [Path, pd.DataFrame], *, is_begin_end: bool = False, as_set: bool = False
    ) -> Iterable[pd.Timestamp]:
        if not isinstance(file_or_df, pd.DataFrame):
            df = self._get_source_data(file_or_df)
        else:
            df = file_or_df
        if df.empty or self.date_field_name not in df.columns.tolist():
            _calendars = pd.Series(dtype=np.float32)
        else:
            _calendars = df[self.date_field_name]

        if is_begin_end and as_set:
            return (_calendars.min(), _calendars.max()), set(_calendars)
        elif is_begin_end:
            return _calendars.min(), _calendars.max()
        elif as_set:
            return set(_calendars)
        else:
            return _calendars.tolist()

    def _get_source_data(self, file_path: Path) -> pd.DataFrame:
        # df = pd.read_csv(str(file_path.resolve()), low_memory=False)
        # df[self.date_field_name] = df[self.date_field_name].astype(str).astype(np.datetime64)
        # # df.drop_duplicates([self.date_field_name], inplace=True)
        # return df

        if 'csv' in file_path.name:
            df = pd.read_csv(str(file_path.resolve()), low_memory=False)
            if 'paused' in df.columns:
                pass
            else:
                df['paused'] = 0
        elif 'h5' in file_path.name:
            df = pd.read_hdf(str(file_path.resolve()))
            df.index.names = ['instrument', 'date']
            df = df.reset_index()
            df['instrument'] = df['instrument'].apply(format_ticker_to_instrument)
            if 'paused' in df.columns:
                pass
            else:
                df['paused'] = 0
            # if 'num_trades' in df.columns:
            #     df = df.drop('num_trades', axis=1)
        df[self.date_field_name] = df[self.date_field_name].astype(str).astype(np.datetime64)
        return df

    def get_symbol_from_file(self, file_path: Path) -> str:
        # return fname_to_code(file_path.name[: -len(self.file_suffix)].strip().lower())
        if self.file_suffix == '.csv':
            return fname_to_code(file_path.name[: -len(self.file_suffix)].strip().lower())
        elif self.file_suffix == '.h5':
            return format_ticker_to_instrument(
                extract_strings(file_path.name)[0])

    def get_dump_fields(self, df_columns: Iterable[str]) -> Iterable[str]:
        return (
            self._include_fields
            if self._include_fields
            else set(df_columns) - set(self._exclude_fields)
            if self._exclude_fields
            else df_columns
        )

    @staticmethod
    def _read_calendars(calendar_path: Path) -> List[pd.Timestamp]:
        return sorted(
            map(
                pd.Timestamp,
                pd.read_csv(calendar_path, header=None).loc[:, 0].tolist(),
            )
        )

    def _read_instruments(self, instrument_path: Path) -> pd.DataFrame:
        df = pd.read_csv(
            instrument_path,
            sep=self.INSTRUMENTS_SEP,
            names=[
                self.symbol_field_name,
                self.INSTRUMENTS_START_FIELD,
                self.INSTRUMENTS_END_FIELD,
            ],
        )

        return df

    def save_calendars(self, calendars_data: list):
        self._calendars_dir.mkdir(parents=True, exist_ok=True)
        calendars_path = str(self._calendars_dir.joinpath(f"{self.freq}.txt").expanduser().resolve())
        result_calendars_list = list(map(lambda x: self._format_datetime(x), calendars_data))
        np.savetxt(calendars_path, result_calendars_list, fmt="%s", encoding="utf-8")

    def save_instruments(self, instruments_data: Union[list, pd.DataFrame]):
        self._instruments_dir.mkdir(parents=True, exist_ok=True)
        instruments_path = str(self._instruments_dir.joinpath(self.INSTRUMENTS_FILE_NAME).resolve())
        if isinstance(instruments_data, pd.DataFrame):
            _df_fields = [self.symbol_field_name, self.INSTRUMENTS_START_FIELD, self.INSTRUMENTS_END_FIELD]
            instruments_data = instruments_data.loc[:, _df_fields]
            instruments_data[self.symbol_field_name] = instruments_data[self.symbol_field_name].apply(
                lambda x: fname_to_code(x.lower()).upper()
            )
            instruments_data.to_csv(instruments_path, header=False, sep=self.INSTRUMENTS_SEP, index=False)
        else:
            np.savetxt(instruments_path, instruments_data, fmt="%s", encoding="utf-8")

    def data_merge_calendar(self, df: pd.DataFrame, calendars_list: List[pd.Timestamp]) -> pd.DataFrame:
        # calendars
        calendars_df = pd.DataFrame(data=calendars_list, columns=[self.date_field_name])
        # calendars_df[self.date_field_name] = calendars_df[self.date_field_name].astype(np.datetime64)
        calendars_df[self.date_field_name] = pd.to_datetime(calendars_df[self.date_field_name])
        # calendars_df[self.date_field_name] = calendars_df[self.date_field_name].astype(np.datetime64[ns])
        cal_df = calendars_df[
            (calendars_df[self.date_field_name] >= df[self.date_field_name].min())
            & (calendars_df[self.date_field_name] <= df[self.date_field_name].max())
        ]
        # align index
        cal_df.set_index(self.date_field_name, inplace=True)
        df.set_index(self.date_field_name, inplace=True)
        r_df = df.reindex(cal_df.index)
        return r_df

    @staticmethod
    def get_datetime_index(df: pd.DataFrame, calendar_list: List[pd.Timestamp]) -> int:
        return calendar_list.index(df.index.min())

    def _data_to_bin(self, df: pd.DataFrame, calendar_list: List[pd.Timestamp], features_dir: Path):
        if df.empty:
            logger.warning(f"{features_dir.name} data is None or empty")
            return
        if not calendar_list:
            logger.warning("calendar_list is empty")
            return
        # align index
        _df = self.data_merge_calendar(df, calendar_list)
        if _df.empty:
            logger.warning(f"{features_dir.name} data is not in calendars")
            return
        # used when creating a bin file
        date_index = self.get_datetime_index(_df, calendar_list)
        for field in self.get_dump_fields(_df.columns):
            bin_path = features_dir.joinpath(f"{field.lower()}.{self.freq}{self.DUMP_FILE_SUFFIX}")
            if field not in _df.columns:
                continue
            if bin_path.exists() and self._mode == self.UPDATE_MODE:
                # update
                with bin_path.open("ab") as fp:
                    np.array(_df[field]).astype("<f").tofile(fp)
            else:
                # append; self._mode == self.ALL_MODE or not bin_path.exists()
                np.hstack([date_index, _df[field]]).astype("<f").tofile(str(bin_path.resolve()))

    def _dump_bin(self, file_or_data: [Path, pd.DataFrame], calendar_list: List[pd.Timestamp]):
        if not calendar_list:
            logger.warning("calendar_list is empty")
            return
        if isinstance(file_or_data, pd.DataFrame):
            if file_or_data.empty:
                return
            code = fname_to_code(str(file_or_data.iloc[0][self.symbol_field_name]).lower())
            df = file_or_data
        elif isinstance(file_or_data, Path):
            code = self.get_symbol_from_file(file_or_data)
            df = self._get_source_data(file_or_data)
        else:
            raise ValueError(f"not support {type(file_or_data)}")
        if df is None or df.empty:
            logger.warning(f"{code} data is None or empty")
            return
        # try to remove dup rows or it will cause exception when reindex.
        df = df.drop_duplicates(self.date_field_name)

        # features save dir
        features_dir = self._features_dir.joinpath(code_to_fname(code).lower())
        features_dir.mkdir(parents=True, exist_ok=True)
        self._data_to_bin(df, calendar_list, features_dir)

    @abc.abstractmethod
    def dump(self):
        raise NotImplementedError("dump not implemented!")

    def __call__(self, *args, **kwargs):
        self.dump()

class DumpDataAll(DumpDataBase):

    def _get_all_date(self):
        logger.info("start get all date......")
        all_datetime = set()
        date_range_list = []
        _fun = partial(self._get_date, as_set=True, is_begin_end=True)
        with tqdm(total=len(self.csv_files)) as p_bar:
            with ProcessPoolExecutor(max_workers=self.works) as executor:
                for file_path, ((_begin_time, _end_time), _set_calendars) in zip(
                    self.csv_files, executor.map(_fun, self.csv_files)
                ):
                    all_datetime = all_datetime | _set_calendars
                    if isinstance(_begin_time, pd.Timestamp) and isinstance(_end_time, pd.Timestamp):
                        _begin_time = self._format_datetime(_begin_time)
                        _end_time = self._format_datetime(_end_time)
                        symbol = self.get_symbol_from_file(file_path)
                        _inst_fields = [symbol.upper(), _begin_time, _end_time]
                        date_range_list.append(f"{self.INSTRUMENTS_SEP.join(_inst_fields)}")
                    p_bar.update()
        self._kwargs["all_datetime_set"] = all_datetime
        self._kwargs["date_range_list"] = date_range_list
        logger.info("end of get all date.\n")

    def _dump_calendars(self):
        logger.info("start dump calendars......")
        self._calendars_list = sorted(map(pd.Timestamp, self._kwargs["all_datetime_set"]))
        self.save_calendars(self._calendars_list)
        logger.info("end of calendars dump.\n")

    def save_calendars(self, calendars_data: list):

        create_table_query = '''
        CREATE TABLE IF NOT EXISTS day (
            datetime String
        ) ENGINE = MergeTree()
        PRIMARY KEY datetime
        ORDER BY datetime
        '''
        CLIENT.command(create_table_query)
        result_calendars_list = list(map(lambda x: self._format_datetime(x), calendars_data))
        CLIENT.insert_df('day', pd.DataFrame(result_calendars_list, columns=['datetime']))


    def _dump_instruments(self):
        logger.info("start dump instruments......")
        self.save_instruments(self._kwargs["date_range_list"])
        logger.info("end of instruments dump.\n")

    def save_instruments(self, instruments_data: Union[list, pd.DataFrame]):
        # self._instruments_dir.mkdir(parents=True, exist_ok=True)
        # pd.DataFrame(instruments_data)

        instruments_df = pd.DataFrame([i.split(self.INSTRUMENTS_SEP) for i in instruments_data], columns=['instrument','in_date','out_date'])
        # instruments_df['instruments'] = self.INSTRUMENTS_FILE_NAME.split('.')[0]
        # instruments_df = instruments_df.loc[:, ['instruments', 'instrument', 'in_date', 'out_date']]

        table_name = self.INSTRUMENTS_FILE_NAME.split('.')[0]
        CLIENT.command('DROP TABLE IF EXISTS {}'.format(table_name))

        create_table_query = '''
        CREATE TABLE IF NOT EXISTS {} (
            instrument String,
            in_date String,
            out_date String
        ) ENGINE = MergeTree()
        PRIMARY KEY instrument
        ORDER BY instrument
        '''.format(table_name)
        CLIENT.command(create_table_query)
        CLIENT.insert_df(table_name, instruments_df)


    def _dump_features(self):

        logger.info("start dump features......")
        _dump_func = partial(self._dump_bin, calendar_list=self._calendars_list)
        dfs = []
        with tqdm(total=len(self.csv_files)) as p_bar:
            with ProcessPoolExecutor(max_workers=self.works) as executor:
                for _df in executor.map(_dump_func, self.csv_files):
                    p_bar.update()
                    dfs.append(_df)

        data = pd.concat(dfs)
        data['paused'] = data['paused'].fillna(0).astype(int)

        create_table_query = '''
        CREATE TABLE IF NOT EXISTS features (
            instrument String,
            date String,
            open Float32,
            close Float32,
            high Float32,
            low Float32,
            vwap Float32,
            volume Float32,
            total_turnover Float32,
            factor Float32,
            paused Int8
        ) ENGINE = MergeTree()
        PRIMARY KEY (instrument, date)
        ORDER BY (instrument, date)
        '''
        CLIENT.command(create_table_query)

        try:
            CLIENT.insert_df('features', data)
        except Exception as e:
            print(e)

        logger.info("end of features dump.\n")

    def _dump_bin(self, file_or_data: [Path, pd.DataFrame], calendar_list: List[pd.Timestamp]):
        if not calendar_list:
            logger.warning("calendar_list is empty")
            return
        if isinstance(file_or_data, pd.DataFrame):
            if file_or_data.empty:
                return
            code = fname_to_code(str(file_or_data.iloc[0][self.symbol_field_name]).lower())
            df = file_or_data
        elif isinstance(file_or_data, Path):
            code = self.get_symbol_from_file(file_or_data)
            df = self._get_source_data(file_or_data)
        else:
            raise ValueError(f"not support {type(file_or_data)}")
        if df is None or df.empty:
            logger.warning(f"{code} data is None or empty")
            return
        # try to remove dup rows or it will cause exception when reindex.
        df = df.drop_duplicates(self.date_field_name)

        # features save dir
        features_dir = self._features_dir.joinpath(code_to_fname(code).lower())
        features_dir.mkdir(parents=True, exist_ok=True)
        return self._data_to_bin(df, calendar_list, features_dir)

    def _data_to_bin(self, df: pd.DataFrame, calendar_list: List[pd.Timestamp], features_dir: Path):
        if df.empty:
            logger.warning(f"{features_dir.name} data is None or empty")
            return
        if not calendar_list:
            logger.warning("calendar_list is empty")
            return
        # align index
        _df = self.data_merge_calendar(df, calendar_list)
        if _df.empty:
            logger.warning(f"{features_dir.name} data is not in calendars")
            return
        # used when creating a bin file
        _df.reset_index(inplace=True)
        _df['instrument'] = features_dir.name.upper()
        if 'paused' in _df.columns:
            pass
        else:
            _df['paused'] = 0


        _df['date'] = _df['date'].astype(str)
        col_list = ['instrument',
                     'date',
                     'open',
                     'close',
                     'high',
                     'low',
                     'vwap',
                     'volume',
                     'total_turnover',
                     'factor',
                     'paused']
        _df = _df.loc[:, col_list]

        return _df

        # create_table_query = '''
        # CREATE TABLE IF NOT EXISTS features (
        #     instrument String,
        #     date String,
        #     open Float32,
        #     close Float32,
        #     high Float32,
        #     low Float32,
        #     volume Float32,
        #     total_turnover Float32,
        #     factor Float32,
        #     paused Int8
        # ) ENGINE = MergeTree()
        # PRIMARY KEY (instrument, date)
        # ORDER BY (instrument, date)
        # '''
        # CLIENT.command(create_table_query)
        # try:
        #     CLIENT.insert_df('features', _df)
        # except Exception as e:
        #     print(f'{features_dir.name.upper()} now saved, please check data: {e}')



        # for field in self.get_dump_fields(_df.columns):
        #     bin_path = features_dir.joinpath(f"{field.lower()}.{self.freq}{self.DUMP_FILE_SUFFIX}")
        #     if field not in _df.columns:
        #         continue
        #     if bin_path.exists() and self._mode == self.UPDATE_MODE:
        #         # update
        #         with bin_path.open("ab") as fp:
        #             np.array(_df[field]).astype("<f").tofile(fp)
        #     else:
        #         # append; self._mode == self.ALL_MODE or not bin_path.exists()
        #         np.hstack([date_index, _df[field]]).astype("<f").tofile(str(bin_path.resolve()))

    def dump(self):
        self._get_all_date()
        self._dump_calendars()
        self._dump_instruments()
        self._dump_features()



if __name__ == '__main__':

    dp = DumpDataAll(csv_path="F:\qlib_data\qlib_csv\qlib_csv_quote", qlib_dir="D:\qlib_data\qlib_cn_db")
    dp.dump()




