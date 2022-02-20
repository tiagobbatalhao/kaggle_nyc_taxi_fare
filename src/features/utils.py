import pandas
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class Transformer(BaseEstimator, TransformerMixin):
    def __init__(self, *args, **kwargs):
        pass

    def fit(self, *args, **kwargs):
        return self

    def fit_transform(self, *args, **kwargs):
        return self.transform(*args, **kwargs)


class FareAmount(Transformer):
    def transform(self, path_iterable, *args, **kwargs):
        df_out = pandas.concat(
            (pandas.read_parquet(p)[["fare_amount"]] for p in path_iterable)
        )
        as_array = df_out.sort_index().values
        return as_array


class PassengerCount(Transformer):
    def transform(self, path_iterable, *args, **kwargs):
        df_out = pandas.concat(
            (pandas.read_parquet(p)[["passenger_count"]] for p in path_iterable)
        )
        as_array = df_out.sort_index().values
        return as_array


class GeographicalBoundingBox(Transformer):
    @staticmethod
    def transform_single(df):
        bounds_lat = (40, 42)
        bounds_lon = (-75, -72)
        flter = (
            (df["pickup_latitude"] > bounds_lat[0])
            & (df["pickup_latitude"] < bounds_lat[1])
            & (df["pickup_longitude"] > bounds_lon[0])
            & (df["pickup_longitude"] < bounds_lon[1])
            & (df["dropoff_latitude"] > bounds_lat[0])
            & (df["dropoff_latitude"] < bounds_lat[1])
            & (df["dropoff_longitude"] > bounds_lon[0])
            & (df["dropoff_longitude"] < bounds_lon[1])
        )
        df["is_correct"] = flter.astype(float)
        return df[["is_correct"]]

    def transform(self, path_iterable, *args, **kwargs):
        df_out = pandas.concat(
            (self.transform_single(pandas.read_parquet(p)) for p in path_iterable)
        )
        as_array = df_out.sort_index().values
        return as_array


class GeographicalCoordinates(Transformer):
    @staticmethod
    def transform_single(df):
        columns = [
            "pickup_latitude",
            "pickup_longitude",
            "dropoff_latitude",
            "dropoff_longitude",
        ]
        bounds_lat = (-90, 90)
        bounds_lon = (-180, 180)
        flter = (
            (df["pickup_latitude"] > bounds_lat[0])
            & (df["pickup_latitude"] < bounds_lat[1])
            & (df["pickup_longitude"] > bounds_lon[0])
            & (df["pickup_longitude"] < bounds_lon[1])
            & (df["dropoff_latitude"] > bounds_lat[0])
            & (df["dropoff_latitude"] < bounds_lat[1])
            & (df["dropoff_longitude"] > bounds_lon[0])
            & (df["dropoff_longitude"] < bounds_lon[1])
        )
        df.loc[~flter, columns] = None
        return df[columns]

    def transform(self, path_iterable, *args, **kwargs):
        df_out = pandas.concat(
            (self.transform_single(pandas.read_parquet(p)) for p in path_iterable)
        )
        as_array = df_out.sort_index().values
        return as_array


class GeographicalDistance(Transformer):
    def transform(self, as_array, *args, **kwargs):
        deg_to_rad = np.pi / 180
        lat1 = as_array[:, 0] * deg_to_rad
        lon1 = as_array[:, 1] * deg_to_rad
        lat2 = as_array[:, 2] * deg_to_rad
        lon2 = as_array[:, 3] * deg_to_rad
        distance = (
            6371
            * 2
            * np.arcsin(
                np.sqrt(
                    np.sin((lat1 - lat2) / 2) ** 2
                    + np.cos(lat1) * np.cos(lat2) * np.sin((lon1 - lon2) / 2) ** 2
                )
            )
        )
        return distance.reshape(-1, 1)


class Timestamp_Week(Transformer):
    @staticmethod
    def transform_single(df):
        day = df["pickup_datetime"].dt.floor("d")
        weekday = df["pickup_datetime"].dt.strftime("%w").astype(int)
        df["time_in_week"] = weekday * 24 + (
            (df["pickup_datetime"] - day).dt.total_seconds() / 3600
        )
        return df[["time_in_week"]]

    def transform(self, path_iterable, *args, **kwargs):
        df_out = pandas.concat(
            (self.transform_single(pandas.read_parquet(p)) for p in path_iterable)
        )
        as_array = df_out.sort_index().values
        return as_array


class PrimaryKey(Transformer):
    @staticmethod
    def transform_single(df):
        df_out = df[[]]
        df_out["index"] = df_out.index
        return df_out

    def transform(self, path_iterable, *args, **kwargs):
        df_out = pandas.concat(
            (self.transform_single(pandas.read_parquet(p)) for p in path_iterable)
        )
        as_array = df_out.sort_index().values
        return as_array


class FourierSeries(Transformer):
    def __init__(self, period, max_freq, *args, **kwargs):
        self.period = period
        self.max_freq = max_freq

    def transform(self, as_array, *args, **kwargs):
        output = []
        angle = as_array.reshape(-1) * 2 * np.pi / self.period
        for freq in range(1, 1 + self.max_freq):
            output.append(np.cos(angle * freq))
            output.append(np.sin(angle * freq))
        return np.array(output).T


class Location(Transformer):
    @staticmethod
    def transform_single(df):
        columns = list(df.columns)
        column_pickup = [x for x in columns if x.startswith("pickup")]
        column_dropoff = [x for x in columns if x.startswith("dropoff")]
        assert len(column_pickup) == 1
        assert len(column_dropoff) == 1
        return df[[column_pickup[0], column_dropoff[0]]]

    def transform(self, path_iterable, df_index=None, *args, **kwargs):
        df_out = pandas.concat(
            (self.transform_single(pandas.read_parquet(p)) for p in path_iterable)
        )
        as_array = df_out.sort_index().values
        return as_array
