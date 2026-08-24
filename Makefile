.PHONY: all data features train evaluate explain verify serve seedstudy labelnoise ieee clean test

all: data features train evaluate explain verify

data:
	python3 -m ringfence.cli data

features:
	python3 -m ringfence.cli features

train:
	python3 -m ringfence.cli train

evaluate:
	python3 -m ringfence.cli evaluate

explain:
	python3 -m ringfence.cli explain

verify:
	python3 -m ringfence.cli verify

serve:
	python3 -m ringfence.cli serve

# Opt-in studies: each refits many models, so neither is part of `all`.
seedstudy:
	python3 -m ringfence.cli seedstudy

labelnoise:
	python3 -m ringfence.cli labelnoise

# The real-data validation. Needs the Kaggle files in data/raw/ieee/ first.
ieee:
	python3 -m ringfence.cli --config configs/ieee_cis.yaml all
	python3 -m ringfence.cli --config configs/ieee_cis.yaml seedstudy

test:
	python3 -m pytest tests -q

clean:
	rm -rf data/*/ reports/*/ site/data/
