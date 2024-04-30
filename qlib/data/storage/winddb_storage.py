"""
Author: Hugo
Date: 2024-04-24 11:24:07
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2024-04-24 15:44:45
Description: qlib读取windDB
"""

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import struct
from pathlib import Path
from typing import Iterable, Union, Dict, Mapping, Tuple, List
from urllib.parse import urlparse
import numpy as np
import pandas as pd

from qlib.utils.time import Freq
from qlib.utils.resam import resam_calendar
from qlib.config import C
from qlib.data.cache import H
from qlib.log import get_module_logger
from qlib.data.storage import (
    CalendarStorage,
    InstrumentStorage,
    FeatureStorage,
    CalVT,
    InstKT,
    InstVT,
)

from sqlalchemy import create_engine

# import psycopg2

logger = get_module_logger("db_storage")


def mapping_table(table_name: str) -> str:
    if table_name.startswith(f"calendars"):

        if table_name.endswith("day"):
            return "ASHARECALENDAR"  # 交易日历表
        else:
            raise ValueError(f"table_name {table_name} not exists")
    elif table_name.startswith(f"instruments"):
        return "ASHAREDESCRIPTION"  # 证券信息表
    elif table_name.startswith("feature"):
        return "ASHAREEODPRICES"  # 行情数据表
    else:
        raise ValueError(f"table_name {table_name} not exists")


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
            engine = create_engine(self.database_uri)
            # print(sql)
            df = pd.read_sql(sql, engine)
            engine.dispose()
            # print(df)
        except Exception as e:
            pass
        # except Exception as e:
        #     print(e)
        return df

    @property
    def db_name(self) -> str:
        res = urlparse(self.database_uri)
        return res.path.lstrip("/")

    @property
    def database_uri(self):
        return (
            C["database_uri"]
            if getattr(self, "_database_uri", None) is None
            else self._database_uri
        )

    @property
    def support_freq(self) -> List[str]:
        _v = "_support_freq"
        if hasattr(self, _v):
            return getattr(self, _v)
        sql = f"SHOW TABLES IN {self.db_name} LIKE 'ASHARECALENDAR'"
        df = self.query(sql)
      
        # print(df['table_name'].tolist())
        if df.empty:
            freq_l = []
        else:
            freq_l = ["day"]
        freq_l = [Freq(freq) for freq in freq_l]
        setattr(self, _v, freq_l)
        return freq_l

    @property
    def uri(self) -> Path:
        return self.table_name

    def exists(self, table_name):
        
        table_name = mapping_table(table_name)
        sql = f"""SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = '{self.db_name}' AND table_name = '{table_name}');"""
        df = self.query(sql)
        return df.iat[0, 0]

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
        table_name: str = mapping_table(self.table_name)
        sql = """SELECT date(TRADE_DAYS)as time FROM {} WHERE S_INFO_EXCHMARKET='SSE' ORDER BY time ASC""".format(
            table_name
        )
        df = self.query(sql)
        if df.empty:
            return []

        return df["time"].tolist()

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
                    raise ValueError(
                        f"can't find a freq from {self.support_freq} that can resample to {self.freq}!"
                    )
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
                np.array(list(map(pd.Timestamp, _calendar))),
                self._freq_db,
                self.freq,
                self.region,
            )
        return _calendar

    def _get_storage_freq(self) -> List[str]:
        return sorted(
            set(map(lambda x: x.stem.split("_")[0], self.uri.parent.glob("*.txt")))
        )

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

    def __setitem__(
        self, i: Union[int, slice], values: Union[CalVT, Iterable[CalVT]]
    ) -> None:
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
        table_name: str = mapping_table(self.table_name)
        sql = (
            f"select S_INFO_WINDCODE as instrument,date(S_INFO_LISTDATE) as start_time,date(S_INFO_DELISTDATE) as end_time from {table_name}"
            " WHERE (S_INFO_WINDCODE LIKE '0%SH' OR S_INFO_WINDCODE LIKE '0%SZ' OR S_INFO_WINDCODE LIKE '6%SH' OR S_INFO_WINDCODE LIKE '6%SZ' OR S_INFO_WINDCODE LIKE '3%SH' OR S_INFO_WINDCODE LIKE '3%SZ')"
        )

        df = self.query(sql)
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
    def __init__(
        self,
        instrument: str,
        field: str,
        freq: str,
        provider_uri: dict = None,
        **kwargs,
    ):
        super(DBFeatureStorage, self).__init__(instrument, field, freq, **kwargs)
        self._provider_uri = (
            None
            if provider_uri is None
            else C.DataPathManager.format_provider_uri(provider_uri)
        )
        self.table_name = f"{self.storage_name}s.{instrument}.{freq}"
        self.table_name = self.table_name.lower()
        self.instrument = instrument.upper()
        self.field = field  # .lower()
        self.freq = freq.lower()
        self.calendar = self.query_calendar()

        self.has_field = self.exists(self.table_name) and self.field_exists()
        self.storage_start_index = self.start_index
        self.storage_end_index = self.end_index

    def query_calendar(self):
        sql = """SELECT date(TRADE_DAYS)as time FROM ASHARECALENDAR WHERE S_INFO_EXCHMARKET='SSE' ORDER BY time ASC"""
        return self.query(sql)["time"].tolist()

    def field_exists(self):

        table_name: str = mapping_table(self.storage_name)

        sql: str = (
            f"select count({self.field}) from {table_name} where S_INFO_WINDCODE='{self.instrument}'"
        )

        df = self.query(sql)
        if df.empty:
            return False
        return df.iloc[0, 0] != 0

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

        table_name = mapping_table(self.storage_name)
        sql = f"""SELECT date(min(TRADE_DT)) FROM {table_name} where S_INFO_WINDCODE = '{self.instrument}';"""

        df = self.query(sql)

        if df.empty:
            return None
        else:
            return self.calendar.index(df.iat[0, 0])

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
                "%Y%m%d"
            )
            table_name = mapping_table(self.storage_name)
            sql = f"select {self.field} from {table_name} where S_INFO_WINDCODE='{self.instrument}' and TRADE_DT={watch_dt};"

            df = self.query(sql)

            self.engine.dispose()
            if df.empty:
                return i, None
            else:
                return i, df.loc[0][0]

        elif isinstance(i, slice):
            start_index = self.storage_start_index if i.start is None else i.start
            end_index = self.storage_end_index if i.stop is None else i.stop - 1
            si = max(start_index, self.storage_start_index)
            if si > end_index:
                return pd.Series(dtype=np.float32)

            # start_id = si - self.storage_start_index + 1
            # end_id = end_index - self.storage_start_index + 1
            start_dt = self.calendar[si].strftime("%Y%m%d")
            end_dt = self.calendar[end_index].strftime("%Y%m%d")
            table_name = mapping_table(self.storage_name)

            sql = f"""SELECT {self.field} FROM {table_name} where S_INFO_WINDCODE='{self.instrument}' and TRADE_DT between {start_dt} and {end_dt};"""

            df = self.query(sql)
            data = df[self.field].to_list()
            # print(si, self.storage_start_index, self.storage_end_index,count)
            series = pd.Series(data, index=pd.RangeIndex(si, si + len(data)))
            # print(series)
            return series
        else:
            raise TypeError(f"type(i) = {type(i)}")

    def __len__(self) -> int:

        table_name = mapping_table(self.storage_name)
        sql = f"select count(*) from {table_name} where S_INFO_WINDCODE='{self.instrument}'"

        df = self.query(sql)
        if df.empty:
            return None
        else:
            return df.loc[0][0] - 1
