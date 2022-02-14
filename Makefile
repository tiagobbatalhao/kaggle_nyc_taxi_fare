parse_data:
	@unzip -uo data/*.zip -d data/
	@split -l 1000000 -d --additional-suffix=.csv data/train.csv data/parts_train
	@rm data/train.csv
	@python src/load_data.py
	@rm -f data/*train*.csv
	@rm -f data/*test*.csv