.PHONY: test train

export PYTHONPATH := src

test:
	pytest -q

train:
	python -m churnlab generate
	python -m churnlab train
	python -m churnlab score
