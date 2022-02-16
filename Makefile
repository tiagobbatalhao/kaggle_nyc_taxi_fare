DATA_RAW = data/raw
DATA_REFINED = data/refined
DATA_EXTERNAL = data/external

parse_data:
	@mkdir -p ${DATA_RAW}
	@unzip -uo data/*.zip -d ${DATA_RAW}
	@split -l 1000000 -d --additional-suffix=.csv ${DATA_RAW}/train.csv ${DATA_RAW}/parts_train
	@rm ${DATA_RAW}/train.csv
	@python src/load_data.py
	@rm -f ${DATA_RAW}/*train*.csv
	@rm -f ${DATA_RAW}/*test*.csv

process_hexagons:
	@mkdir -p ${DATA_REFINED}
	@python src/features/coordinates.py --feature hexagon --folder ${DATA_REFINED}

download_geojson:
	@mkdir -p ${DATA_EXTERNAL}
	@python src/external/geography.py --folder ${DATA_EXTERNAL}
