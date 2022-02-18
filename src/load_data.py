import os
import pandas
import numpy as np
from datetime import datetime
import glob
import logging
import hashlib

DATA_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))


def _parse_data(file_obj, type_, header, limit_lines=None):
    read_kwargs = {}
    read_kwargs["header"] = header
    read_kwargs["dtype"] = {
        "key": "str",
        "fare_amount": np.float32,
        "pickup_datetime": "str",
        "pickup_longitude": np.float32,
        "pickup_latitude": np.float32,
        "dropoff_longitude": np.float32,
        "dropoff_latitude": np.float32,
        "passenger_count": np.int32,
    }
    if isinstance(limit_lines, int) and (limit_lines > 0):
        read_kwargs["nrows"] = limit_lines

    df = pandas.read_csv(file_obj, **read_kwargs)

    columns = [
        "key",
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ]
    if type_ == "train":
        columns.insert(1, "fare_amount")

    if header is None:
        df.columns = columns
    df = df[columns]

    column_timestamp = "pickup_datetime"
    if column_timestamp in df.columns:
        df[column_timestamp] = df[column_timestamp].apply(
            lambda s: datetime.strptime(s[:19], "%Y-%m-%d %H:%M:%S")
        )

    df.index = (
        df["key"]
        .apply(lambda x: int(hashlib.md5(x.encode("utf8")).hexdigest()[:16], base=16))
        .values
    )
    return df


def convert_to_parquet():
    files = glob.glob(os.path.join(DATA_FOLDER, "raw", "parts_train*.csv"))
    files += glob.glob(os.path.join(DATA_FOLDER, "raw", "test*.csv"))
    for fl in sorted(files):
        logging.info("Loading {}".format(fl))
        if "train" in fl:
            if "00.csv" in fl:
                kwargs = {"type_": "train", "header": 0}
            else:
                kwargs = {"type_": "train", "header": None}
        else:
            kwargs = {"type_": "test", "header": 0}
        with open(fl, "r") as f:
            df = _parse_data(f, **kwargs)
        fl_save = fl.replace(".csv", ".parquet")
        df.to_parquet(fl_save)
        logging.info("Saved {}".format(fl_save))


def get_data_files():
    files_train = glob.glob(os.path.join(DATA_FOLDER, "raw", "*train*.parquet"))
    files_test = glob.glob(os.path.join(DATA_FOLDER, "raw", "*test*.parquet"))
    return sorted(files_train), sorted(files_test)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    convert_to_parquet()
