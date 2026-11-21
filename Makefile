.PHONY: install tune backtest train app test all
install:
	pip install -r requirements-dev.txt
tune:
	python tune.py
backtest:
	python backtest.py
train:
	python train.py
app:
	streamlit run app.py
test:
	pytest -q
all: tune backtest train
