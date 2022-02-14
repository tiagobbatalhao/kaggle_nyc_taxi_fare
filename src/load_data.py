import zipfile
import os
import pandas
import numpy as np
from datetime import datetime

def _load_zip(path):
    zipped = zipfile.ZipFile(path)
    for name in zipped.namelist():
        with zipped.open(name) as f:
            yield name, f

def _parse_data(file_obj, limit_lines=None):
    read_kwargs = {}
    read_kwargs['dtype'] = {
        'key': 'str',
        'fare_amount': np.float32,
        'pickup_datetime': 'str',
        'pickup_longitude': np.float32,
        'pickup_latitude': np.float32,
        'dropoff_longitude': np.float32,
        'dropoff_latitude': np.float32,
        'passenger_count': np.int32,
    }
    if isinstance(limit_lines, int) and (limit_lines > 0):
        read_kwargs['nrows'] = limit_lines
    df = pandas.read_csv(file_obj, **read_kwargs)
    print("LOADED: {} rows".format(len(df)))

    column_timestamp = 'pickup_datetime'
    if column_timestamp in df.columns:
        df[column_timestamp] = df[column_timestamp].apply(
            lambda s: datetime.strptime(s[:19], '%Y-%m-%d %H:%M:%S')
        )
    print("TRANSFORMED TO TIMESTAMP")
    return df

def load_data(force=False, **kwargs):
    this_folder = os.path.dirname(__file__)
    data_folder = os.path.join(this_folder, '..', 'data')
    
    path_train = os.path.join(data_folder, 'train.parquet')
    path_test = os.path.join(data_folder, 'test.parquet')
    if (not force) and (os.path.exists(path_train)) and(os.path.exists(path_test)):
        df_train = pandas.read_parquet(path_train)
        df_test = pandas.read_parquet(path_test)
        return df_train, df_test
    else:
        datasets = {}
        filename = "new-york-city-taxi-fare-prediction.zip"
        path = os.path.abspath(os.path.join(data_folder, filename))
        for name, obj in _load_zip(path):
            print("READING {}".format(name))
            if name == "train.csv":
                datasets['train'] = _parse_data(obj, limit_lines=kwargs.get('limit_lines'))
                datasets['train'].to_parquet(path_train)
            elif name == "test.csv":
                datasets['test'] = _parse_data(obj)
                datasets['test'].to_parquet(path_test)
        return datasets.get('train'), datasets.get('test')

