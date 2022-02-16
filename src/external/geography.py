import functools
import logging
from h3 import h3
import requests
import json
import os
import argparse
import geopandas
import io
import pandas


def download_geojson(folder_output, datasets=None):
    saved_files = []
    if datasets is not None:
        datasets = [x.strip() for x in datasets.split(",")]
    this_folder = os.path.dirname(__file__)
    with open(os.path.join(this_folder, "geography_links.json")) as f:
        links = json.load(f)
    for elem in links:
        if (datasets is None) or (elem["name"] in datasets):
            req = requests.get(elem["link"])
            if req.status_code == 200:
                fl_save = os.path.join(
                    folder_output,
                    "{}_{}.geojson".format(elem["name"], elem["boundary"]),
                )
                with open(fl_save, "w") as f:
                    f.write(req.text)
                logging.info("Saved {}".format(fl_save))
                saved_files.append(fl_save)
    return saved_files


def shape_to_hexagons(shape, resolution):
    if shape.type == "Polygon":
        hexagons = h3.compact(
            h3.polyfill(
                shape.__geo_interface__,
                resolution, geo_json_conformant=True,
            )
        )
    elif shape.type == "MultiPolygon":
        hexagons = []
        for g in shape.geoms:
            hexagons += shape_to_hexagons(g, resolution)
    return sorted(hexagons)


def convert_hexagon(geojson, resolution):
    with io.StringIO(json.dumps(geojson)) as f:
        dfgeo = geopandas.read_file(f)
    dfgeo["hexagons"] = dfgeo["geometry"].apply(
        functools.partial(shape_to_hexagons, resolution=resolution)
    )
    dfgeo = pandas.DataFrame(dfgeo.drop("geometry", axis=1))
    return dfgeo


def process_geometry(folder_output, datasets, resolution=11):
    saved_files = download_geojson(folder_output, datasets)
    for fl in saved_files:
        with open(fl) as f:
            geojson = json.load(f)
        df = convert_hexagon(geojson, resolution)
        fl_save = fl.replace(".geojson", ".parquet")
        df.to_parquet(fl_save)
        logging.info("Saved {}".format(fl_save))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder")
    parser.add_argument("--datasets", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    process_geometry(args.folder, args.datasets)
