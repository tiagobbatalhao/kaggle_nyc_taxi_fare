from h3 import h3
import pandas
import os
import logging
import argparse
import itertools
import glob


from src.load_data import DATA_FOLDER, get_data_files


def get_hexagon(df, prefix, resolution, as_integer=True):
    min_lat, max_lat = -90, +90
    min_lon, max_lon = -180, 180
    flter = (
        (df["pickup_latitude"] > min_lat)
        & (df["pickup_latitude"] < max_lat)
        & (df["pickup_longitude"] > min_lon)
        & (df["pickup_longitude"] < max_lon)
        & (df["dropoff_latitude"] > min_lat)
        & (df["dropoff_latitude"] < max_lat)
        & (df["dropoff_longitude"] > min_lon)
        & (df["dropoff_longitude"] < max_lon)
    )
    column = prefix + "_hexagon"
    df.loc[flter, column] = df.loc[flter].apply(
        lambda row: h3.geo_to_h3(
            row[prefix + "_latitude"], row[prefix + "_longitude"], resolution
        ),
        axis=1,
    )
    if as_integer:
        df[column] = df[column].fillna("0").apply(lambda x: int(x, 16))
    return df


def get_external_shapes(name):
    fl = os.path.join(DATA_FOLDER, "external", name + ".parquet")
    df = pandas.read_parquet(fl)
    return df


def find_location(df_points, df_area, prefix, region_id):
    col = prefix + "_hexagon_ls"
    df_points[col] = df_points[prefix + "_hexagon"].apply(
        lambda s: [int(h3.h3_to_parent(hex(s)[2:], res=r), 16) for r in range(1, 16)]
        if isinstance(s, int) and s > 0
        else []
    )
    dfA = df_area[[region_id, "hexagons"]].explode("hexagons").dropna()
    dfB = (
        df_points.reset_index()[["index", col]]
        .explode(col)
        .rename(columns={col: "hexagons"})
        .dropna()
    )
    dfA["hexagons"] = dfA["hexagons"].astype(int)
    dfB["hexagons"] = dfB["hexagons"].astype(int)
    df = (
        dfA.merge(dfB, on=["hexagons"])[["index", region_id]]
        .rename(columns={region_id: prefix + "_" + region_id})
        .drop_duplicates(subset=["index"])
        .set_index("index")
    )
    df = df_points[[]].join(df, how="left")
    return df


def process_hexagons(folder_output, resolution=15):
    files = get_data_files()
    for fl in itertools.chain(*files):
        df = pandas.read_parquet(fl)
        df = get_hexagon(df, "pickup", resolution)
        df = get_hexagon(df, "dropoff", resolution)
        _, base = os.path.split(fl)
        fl_save = os.path.join(folder_output, "hexagon_" + base)
        df = df.filter(regex="hexagon")
        df.to_parquet(fl_save)
        logging.info("Saved {}".format(fl_save))


def process_location(folder_output, location_type, region_id):
    df_area = get_external_shapes(location_type)
    files = glob.glob(os.path.join(DATA_FOLDER, "refined", "hexagon_*.parquet"))
    for fl in sorted(files):
        df = pandas.read_parquet(fl)
        dfA = find_location(df, df_area, prefix="pickup", region_id=region_id)
        dfB = find_location(df, df_area, prefix="dropoff", region_id=region_id)
        dfsave = df[[]].join(dfA, how="left").join(dfB, how="left")
        _, base = os.path.split(fl)
        fl_save = os.path.join(folder_output, base.replace("hexagon", location_type))
        dfsave.to_parquet(fl_save)
        logging.info("Saved {}".format(fl_save))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature")
    parser.add_argument("--folder")
    parser.add_argument("--location_type", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.feature == "hexagon":
        process_hexagons(args.folder)
    if args.feature == "location":
        region_ids = {
            "borough_shoreline": "BoroName",
            "borough_water": "BoroName",
            "state_assembly_shoreline": "AssemDist",
            "state_assembly_water": "AssemDist",
        }
        region_id = region_ids[args.location_type]
        process_location(args.folder, args.location_type, region_id)
