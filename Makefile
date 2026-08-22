.PHONY: all data features train evaluate verify clean test

all: data features train evaluate verify

data:
	python3 -m ringfence.cli data

features:
	python3 -m ringfence.cli features

train:
	python3 -m ringfence.cli train

evaluate:
	python3 -m ringfence.cli evaluate

verify:
	python3 -m ringfence.cli verify

test:
	python3 -m pytest tests -q

clean:
	rm -rf data/*.parquet data/*.csv.gz reports/*.csv reports/*.json reports/*.pkl
