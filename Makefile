.PHONY: install run test data train warm
install:
	python -m pip install -r requirements-dev.txt
run:
	uvicorn wasteops.api:app --reload --host 127.0.0.1 --port 8000
test:
	pytest -q
data:
	python scripts/download_realwaste.py
train:
	python scripts/train.py
warm:
	python scripts/warm_model.py
