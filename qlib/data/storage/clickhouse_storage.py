# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import struct
from pathlib import Path
from typing import Iterable, Union, Dict, Mapping, Tuple, List

import numpy as np
import pandas as pd

from qlib.utils.time import Freq
from qlib.utils.resam import resam_calendar
from qlib.config import C
from qlib.data.cache import H
from qlib.data.data import Clickhouse
from qlib.log import get_module_logger
from qlib.data.storage import CalendarStorage, InstrumentStorage, FeatureStorage, CalVT, InstKT, InstVT

from sqlalchemy import create_engine
import psycopg2

logger = get_module_logger("db_storage")


class DBStorageMixin:
    """DBStorageMixin, applicable to DBXXXStorage
    Subclasses need to have database_uri, freq, storage_name, file_name attributes
    """

    # NOTE: provider_uri priority:
    #   1. self._provider_uri : if provider_uri is provided.
    #   2. provider_uri in qlib.config.C
    def query(self, sql):
        df = pd.DataFrame()
        try:
            df = Clickhouse.query_df(sql)
        except Exception as e:
            print(e)
        return df

    @property
    def database_uri(self):
        return C["database_uri"] if getattr(self, "_database_uri", None) is None else self._database_uri

    @property
    def support_freq(self) -> List[str]:
        _v = "_support_freq"
        if hasattr(self, _v):
            return getattr(self, _v)
        freq_l = ["day","min"]
        freq_l = [Freq(freq) for freq in freq_l]
        setattr(self, _v, freq_l)
        return freq_l

    @property
    def uri(self) -> Path:
        return self.table_name

    def exists(self, table_name):

        table_name = table_name.split('.')[-1]
        sql = f'''SELECT EXISTS(
            SELECT 1
            FROM system.tables
            WHERE database = 'default' AND name = '{table_name}'
        ) AS table_exists;'''
        df = Clickhouse.query_df(sql)
        return df.iloc[0,0]

    def check(self):
        """check self.uri

        Raises
        -------
        ValueError
        """
        if not self.exists(self.uri):
            raise ValueError(f"{self.storage_name} not exists: {self.uri}")


class DBCalendarStorage(DBStorageMixin, CalendarStorage):
    def __init__(self, freq: str, future: bool, provider_uri: dict = None, **kwargs):
        super(DBCalendarStorage, self).__init__(freq, future, **kwargs)
        self.future = future
        self.enable_read_cache = True  # TODO: make it configurable
        self.region = C["region"]
        self.table_name = f"{self.storage_name}s.{self._freq_db}"
        self.table_name = self.table_name.lower()


    def _read_calendar(self) -> List[CalVT]:
        # NOTE:
        # if we want to accelerate partial reading calendar
        # we can add parameters like `skip_rows: int = 0, n_rows: int = None` to the interface.
        # Currently, it is not supported for the txt-based calendar

        if not self.exists(self.table_name):
            self._write_calendar(values=[])

        sql = '''SELECT date(datetime)as time FROM default.{};'''.format(self.table_name.split('.')[-1])
        df = Clickhouse.query_df(sql)
        if df.empty:
            return []

        return df['time'].tolist()

    def _write_calendar(self, values: Iterable[CalVT], mode: str = "wb"):
        pass

    @property
    def _freq_db(self) -> str:
        """the freq to read from file"""
        if not hasattr(self, "_freq_file_cache"):
            freq = Freq(self.freq)
            if freq not in self.support_freq:
                # NOTE: uri
                #   1. If `uri` does not exist
                #       - Get the `min_uri` of the closest `freq` under the same "directory" as the `uri`
                #       - Read data from `min_uri` and resample to `freq`

                freq = Freq.get_recent_freq(freq, self.support_freq)
                if freq is None:
                    raise ValueError(f"can't find a freq from {self.support_freq} that can resample to {self.freq}!")
            self._freq_file_cache = freq
        return self._freq_file_cache

    @property
    def data(self) -> List[CalVT]:
        # self.check()
        # If cache is enabled, then return cache directly
        if self.enable_read_cache:
            key = "orig_file" + str(self.uri)
            if key not in H["c"]:
                H["c"][key] = self._read_calendar()
            _calendar = H["c"][key]
        else:
            _calendar = self._read_calendar()
        if Freq(self._freq_db) != Freq(self.freq):
            _calendar = resam_calendar(
                np.array(list(map(pd.Timestamp, _calendar))), self._freq_db, self.freq, self.region
            )
        return _calendar

    def _get_storage_freq(self) -> List[str]:
        return sorted(set(map(lambda x: x.stem.split("_")[0], self.uri.parent.glob("*.txt"))))

    def extend(self, values: Iterable[CalVT]) -> None:
        self._write_calendar(values, mode="ab")

    def clear(self) -> None:
        self._write_calendar(values=[])

    def index(self, value: CalVT) -> int:
        self.check()
        calendar = self._read_calendar()
        return int(np.argwhere(calendar == value)[0])

    def insert(self, index: int, value: CalVT):
        calendar = self._read_calendar()
        calendar = np.insert(calendar, index, value)
        self._write_calendar(values=calendar)

    def remove(self, value: CalVT) -> None:
        self.check()
        index = self.index(value)
        calendar = self._read_calendar()
        calendar = np.delete(calendar, index)
        self._write_calendar(values=calendar)

    def __setitem__(self, i: Union[int, slice], values: Union[CalVT, Iterable[CalVT]]) -> None:
        calendar = self._read_calendar()
        calendar[i] = values
        self._write_calendar(values=calendar)

    def __delitem__(self, i: Union[int, slice]) -> None:
        self.check()
        calendar = self._read_calendar()
        calendar = np.delete(calendar, i)
        self._write_calendar(values=calendar)

    def __getitem__(self, i: Union[int, slice]) -> Union[CalVT, List[CalVT]]:
        self.check()
        return self._read_calendar()[i]

    def __len__(self) -> int:
        return len(self.data)


class DBInstrumentStorage(DBStorageMixin, InstrumentStorage):
    INSTRUMENT_SEP = "\t"
    INSTRUMENT_START_FIELD = "start_datetime"
    INSTRUMENT_END_FIELD = "end_datetime"
    SYMBOL_FIELD_NAME = "instrument"

    def __init__(self, market: str, freq: str, provider_uri: dict = None, **kwargs):
        super(DBInstrumentStorage, self).__init__(market, freq, **kwargs)
        self.table_name = f"{self.storage_name}s.{market.lower()}"
        self.table_name = self.table_name.lower()

    def _read_instrument(self) -> Dict[InstKT, InstVT]:
        if not self.exists(self.table_name):
            raise FileNotFoundError(self.table_name)

        _instruments = dict()
        sql = '''select instrument, date(in_date)as start_time, date(out_date)as end_time 
                from default.{}'''.format(self.table_name.split('.')[-1])
        df = Clickhouse.query_df(sql)
        for row in df.itertuples(index=False):
            _instruments.setdefault(row[0], []).append((row[1], row[2]))

        return _instruments

    def _write_instrument(self, data: Dict[InstKT, InstVT] = None) -> None:
        raise NotImplementedError(f"Please use other database tools to write!")

    def clear(self) -> None:
        self._write_instrument(data={})

    @property
    def data(self) -> Dict[InstKT, InstVT]:
        self.check()
        return self._read_instrument()

    def __setitem__(self, k: InstKT, v: InstVT) -> None:
        inst = self._read_instrument()
        inst[k] = v
        self._write_instrument(inst)

    def __delitem__(self, k: InstKT) -> None:
        self.check()
        inst = self._read_instrument()
        del inst[k]
        self._write_instrument(inst)

    def __getitem__(self, k: InstKT) -> InstVT:
        self.check()
        return self._read_instrument()[k]

    def update(self, *args, **kwargs) -> None:
        if len(args) > 1:
            raise TypeError(f"update expected at most 1 arguments, got {len(args)}")
        inst = self._read_instrument()
        if args:
            other = args[0]  # type: dict
            if isinstance(other, Mapping):
                for key in other:
                    inst[key] = other[key]
            elif hasattr(other, "keys"):
                for key in other.keys():
                    inst[key] = other[key]
            else:
                for key, value in other:
                    inst[key] = value
        for key, value in kwargs.items():
            inst[key] = value

        self._write_instrument(inst)

    def __len__(self) -> int:
        return len(self.data)

class DBFeatureStorage(DBStorageMixin, FeatureStorage):
    def __init__(self, instrument: str, field: str, freq: str, provider_uri: dict = None, **kwargs):
        super(DBFeatureStorage, self).__init__(instrument, field, freq, **kwargs)
        self._provider_uri = None if provider_uri is None else C.DataPathManager.format_provider_uri(provider_uri)
        self.table_name = f"{self.storage_name}s.{instrument}.{freq}"
        self.table_name = self.table_name.lower()
        self.table_name = self.table_name.split('.')[0]
        self.instrument = instrument.lower()
        self.field = field.lower()
        self.freq = freq.lower()
        self.calendar = self._read_calendar()

        self.has_field = self.exists(self.table_name) and self.field_exists()
        self.storage_start_index = self.start_index
        self.storage_end_index = self.end_index

    def _read_calendar(self) -> List[CalVT]:
        # NOTE:
        # if we want to accelerate partial reading calendar
        # we can add parameters like `skip_rows: int = 0, n_rows: int = None` to the interface.
        # Currently, it is not supported for the txt-based calendar

        if not self.exists(self.table_name):
            self._write_calendar(values=[])

        sql = '''SELECT date(datetime)as time FROM default.{};'''.format(self.freq)
        df = Clickhouse.query_df(sql)
        if df.empty:
            return []

        return df['time'].tolist()

    def field_exists(self):
        sql = f'''select {self.field} from default.{self.table_name} where instrument = '{self.instrument.upper()}' limit 1;'''
        df = Clickhouse.query_df(sql)
        if df.empty:
            return False
        return len(df) != 0

    def clear(self):
        with self.uri.open("wb") as _:
            pass

    @property
    def data(self) -> pd.Series:
        return self[:]

    def write(self, data_array: Union[List, np.ndarray], index: int = None) -> None:
        raise NotImplementedError(f"Please use other database tools to write!")

    @property
    def start_index(self) -> Union[int, None]:
        if not self.has_field:
            return None
        # sql = '''SELECT {} FROM default.{} limit 1'''.format(self.field, self.table_name)
        sql = f'''select date from default.{self.table_name} where instrument = '{self.instrument.upper()}' order by date limit 1;'''
        df = Clickhouse.query_df(sql)

        if df.empty:
            return None
        else:
            return self.calendar.index(pd.to_datetime(df.iat[0, 0]))
        return 0

    @property
    def end_index(self) -> Union[int, None]:
        if not self.has_field:
            return None

        return self.start_index + len(self) - 1

    def __getitem__(self, i: Union[int, slice]) -> Union[Tuple[int, float], pd.Series]:
        if not self.has_field:
            if isinstance(i, int):
                return None, None
            elif isinstance(i, slice):
                return pd.Series(dtype=np.float32)
            else:
                raise TypeError(f"type(i) = {type(i)}")

        if isinstance(i, int):
            if self.storage_start_index > i:
                raise IndexError(f"{i}: start index is {self.storage_start_index}")

            watch_dt = self.calendar[i - self.storage_start_index + 1].strftime(
                "%Y-%m-%d"
            )

            sql = f'''select instrument, date, {self.field} from default.{self.table_name} where instrument = '{self.instrument.upper()}' and date = '{watch_dt}';'''

            df = Clickhouse.query_df(sql)

            # self.engine.dispose()
            if df.empty:
                return i, None
            else:
                return i, df.iloc[0, 2]

        elif isinstance(i, slice):
            start_index = self.storage_start_index if i.start is None else i.start
            end_index = self.storage_end_index if i.stop is None else i.stop - 1
            si = max(start_index, self.storage_start_index)
            if si > end_index:
                return pd.Series(dtype=np.float32)

            # start_id = si - self.storage_start_index + 1
            # end_id = end_index - self.storage_start_index + 1
            # start_dt = self.calendar[si].strftime("%Y%m%d")
            # end_dt = self.calendar[end_index].strftime("%Y%m%d")

            start_dt = self.calendar[si].strftime(
                "%Y-%m-%d"
            )
            end_dt = self.calendar[end_index].strftime(
                "%Y-%m-%d"
            )

            sql = f'''select instrument, date, {self.field} from default.{self.table_name} where instrument = '{self.instrument.upper()}' and date >= '{start_dt}' and date <= '{end_dt}';'''



            df = Clickhouse.query_df(sql)

            try:
                data = df[self.field].to_list()
            except:
                return pd.Series()
            # print(si, self.storage_start_index, self.storage_end_index,count)
            series = pd.Series(data, index=pd.RangeIndex(si, si + len(data)))
            # print(series)
            return series
        else:
            raise TypeError(f"type(i) = {type(i)}")

    def __len__(self) -> int:
        sql = f'''SELECT count(*) FROM default.{self.table_name} where instrument = '{self.instrument.upper()}';'''
        df = Clickhouse.query_df(sql)
        if df.empty:
            return None
        else:
            return int(df.loc[0][0] - 1)