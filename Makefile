DATA_RAW = data/raw
DATA_REFINED = data/refined
DATA_EXTERNAL = data/external
DATA_FEATURES = data/features

parse_data:
	@mkdir -p ${DATA_RAW}
	@unzip -uo data/*.zip -d ${DATA_RAW}
	@split -l 1000000 -d --additional-suffix=.csv ${DATA_RAW}/train.csv ${DATA_RAW}/parts_train
	@rm ${DATA_RAW}/train.csv
	@python src/load_data.py
	@rm -f ${DATA_RAW}/*train*.csv
	@rm -f ${DATA_RAW}/*test*.csv

download_geojson:
	@mkdir -p ${DATA_EXTERNAL}
	@python src/external/geography.py --folder ${DATA_EXTERNAL}

process_hexagons:
	@mkdir -p ${DATA_REFINED}
	@PYTHONPATH=. python src/features/coordinates.py --feature hexagon --folder ${DATA_REFINED}

process_location:
	@mkdir -p ${DATA_EXTERNAL}
	@PYTHONPATH=. python src/features/coordinates.py --feature location --folder ${DATA_REFINED} --location_type ${LOCATION_TYPE}

features:
	@mkdir -p ${DATA_FEATURES}
	@PYTHONPATH=. python src/features/pre_processing.py --script generate --folder ${DATA_FEATURES}

features_simplify:
	@mkdir -p ${DATA_FEATURES}
	@PYTHONPATH=. python src/features/pre_processing.py --script simplify --folder ${DATA_FEATURES} --examples ${EXAMPLES}

features_train_test_split:
	@mkdir -p ${DATA_FEATURES}
	@PYTHONPATH=. python src/features/pre_processing.py --script train_test_split --folder ${DATA_FEATURES} --examples ${EXAMPLES}
