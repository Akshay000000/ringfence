.PHONY: all data features train evaluate explain verify serve seedstudy labelnoise ieee site clean test ci narrative structure

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

# Is there any collusion structure in this dataset at all? Permutation test.
structure:
	python3 -m ringfence.cli structure

# Draft merchant-facing notes and fact-check them. Template path unless
# RINGFENCE_LLM_KEY is set.
narrative:
	python3 -m ringfence.cli narrative

# The real-data validation. Needs the Kaggle files in data/raw/ieee/ first.
ieee:
	python3 -m ringfence.cli --config configs/ieee_cis.yaml all
	python3 -m ringfence.cli --config configs/ieee_cis.yaml seedstudy

test:
	python3 -m pytest tests -q

# What CI runs: the unit tests, then a tiny end-to-end build. Unit tests cannot
# catch a pipeline that no longer runs, and `make all` is too slow for a hook.
ci:
	python3 -m pytest tests -q
	python3 -m ringfence.cli --config configs/ci_smoke.yaml all
	python3 -m ringfence.cli --config configs/ci_smoke.yaml verify
	python3 -m ringfence.cli --config configs/ci_smoke.yaml narrative

clean:
	rm -rf data/*/ reports/*/ site/data/
