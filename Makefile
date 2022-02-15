DATA_FOLDER = data/raw

parse_data:
	@mkdir -p ${DATA_FOLDER}
	@unzip -uo data/*.zip -d ${DATA_FOLDER}
	@split -l 1000000 -d --additional-suffix=.csv ${DATA_FOLDER}/train.csv ${DATA_FOLDER}/parts_train
	@rm ${DATA_FOLDER}/train.csv
	@python src/load_data.py
	@rm -f ${DATA_FOLDER}/*train*.csv
	@rm -f ${DATA_FOLDER}/*test*.csv
