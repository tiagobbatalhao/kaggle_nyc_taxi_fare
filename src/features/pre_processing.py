import glob
import numpy as np
import pickle
import os
import logging
import argparse

from sklearn.pipeline import Pipeline

import src.load_data
import src.features.utils as ml_utils


def detect_outliers(files):
    good_cordinates = ml_utils.GeographicalBoundingBox().transform(files)
    distance = Pipeline(
        [
            ("coordinates", ml_utils.GeographicalCoordinates()),
            ("distance", ml_utils.GeographicalDistance()),
        ]
    ).transform(files)
    fare = ml_utils.FareAmount().transform(files)
    is_good = (
        (good_cordinates == 1)
        & (distance > 0)
        & (distance < 100)
        & (fare > 0)
        & (fare < 100)
    )
    return is_good.reshape(-1)


def train_test_split_index(is_good, fraction_train):
    size = len(is_good)
    status = (np.random.random(size) > fraction_train).astype(int)
    status[(is_good == 0)] = 2
    return status


def make_coordinates(files):
    coordinates = ml_utils.GeographicalCoordinates().transform(files)
    return coordinates.astype(float)


def make_distance(files):
    distance = Pipeline(
        [
            ("coordinates", ml_utils.GeographicalCoordinates()),
            ("distance", ml_utils.GeographicalDistance()),
        ]
    ).transform(files)
    return distance.astype(float).reshape(-1, 1)


def make_fare(files):
    fare = ml_utils.FareAmount().transform(files)
    return fare.astype(float).reshape(-1, 1)


def make_timestamp_week(files):
    timestamp = ml_utils.Timestamp_Week().transform(files)
    return timestamp.astype(float).reshape(-1, 1)


def make_passenger_count(files):
    passenger_count = ml_utils.PassengerCount().transform(files)
    return passenger_count


def make_primary_key(files):
    primary_key = ml_utils.PrimaryKey().transform(files)
    return primary_key


def make_location(files):
    location = ml_utils.Location().transform(files)
    return location


def generate_features(folder_save):
    files = {}
    files["train"], files["test"] = src.load_data.get_data_files()
    plan = [
        (
            lambda *args: train_test_split_index(detect_outliers(*args), 0.75),
            "train_test_split",
        ),
        (make_fare, "fare"),
        (make_passenger_count, "passenger_count"),
        (make_coordinates, "coordinates"),
        (make_distance, "distance"),
        (make_timestamp_week, "timestamp_week"),
        (make_primary_key, "primary_key"),
    ]
    folder_save = os.path.join(folder_save, "full")
    os.makedirs(folder_save, exist_ok=True)
    for func, name in plan:
        fl_save = os.path.join(folder_save, f"{name}.pickle")
        if os.path.exists(fl_save):
            continue
        arr = func(files["train"])
        with open(fl_save, "wb") as f:
            pickle.dump(arr.reshape(-1, 1), f)
        logging.info(f"Saved {fl_save}")

    folder_refined = os.path.join(src.load_data.DATA_FOLDER, "refined")
    for name in ["borough", "state_assembly"]:
        fl_save = os.path.join(folder_save, f"{name}.pickle")
        if os.path.exists(fl_save):
            continue
        fls = glob.glob(os.path.join(folder_refined, f"{name}*train*.parquet"))
        arr = make_location(fls)
        with open(fl_save, "wb") as f:
            pickle.dump(arr.reshape(-1, 1), f)
        logging.info(f"Saved {fl_save}")


def simplify(folder_data, examples):
    fls = glob.glob(os.path.join(folder_data, "full", "*.pickle"))
    folder_save = os.path.join(folder_data, "small")
    os.makedirs(folder_save, exist_ok=True)
    for fl in fls:
        with open(fl, "rb") as f:
            arr = pickle.load(f)
        _, base = os.path.split(fl)
        fl_save = os.path.join(folder_save, base)
        with open(fl_save, "wb") as f:
            pickle.dump(arr[:examples, :], f)
        logging.info(f"Saved {fl_save}")


def train_test_split(folder_data, examples):
    folder_save_train = os.path.join(folder_data, "train")
    folder_save_test = os.path.join(folder_data, "test")
    os.makedirs(folder_save_train, exist_ok=True)
    os.makedirs(folder_save_test, exist_ok=True)

    with open(os.path.join(folder_data, "full", "train_test_split.pickle"), "rb") as f:
        status = pickle.load(f).reshape(-1)[:examples]
    fls = glob.glob(os.path.join(folder_data, "full", "*.pickle"))
    for fl in fls:
        with open(fl, "rb") as f:
            arr = pickle.load(f)[:examples, :]
        
        _, base = os.path.split(fl)
        fl_save = os.path.join(folder_save_train, base)
        with open(fl_save, "wb") as f:
            pickle.dump(arr[status==0, :], f)
        logging.info(f"Saved {fl_save}")
        fl_save = os.path.join(folder_save_test, base)
        with open(fl_save, "wb") as f:
            pickle.dump(arr[status==1, :], f)
        logging.info(f"Saved {fl_save}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder")
    parser.add_argument("--script")
    parser.add_argument("--examples", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.script == "generate":
        generate_features(args.folder)
    if args.script == "simplify":
        simplify(args.folder, args.examples)
    if args.script == "train_test_split":
        train_test_split(args.folder, args.examples)
