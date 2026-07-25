.PHONY: setup download train evaluate infer-image infer-folder infer-video app test lint clean

setup:
	uv sync

download:
	uv run python main.py download

train:
	uv run python main.py train

train-%:
	uv run python main.py train --model $*

evaluate:
	uv run python main.py evaluate

infer-image:
	uv run python main.py infer-image --input $(INPUT) --output $(OUTPUT)

infer-folder:
	uv run python main.py infer-folder --input $(INPUT) --output $(OUTPUT)

infer-video:
	uv run python main.py infer-video --input $(INPUT) --output $(OUTPUT)

app:
	uv run streamlit run app/app.py

test:
	uv run pytest -v

lint:
	uv run black --check src tests scripts app main.py
	uv run flake8 src tests scripts app main.py

clean:
	rm -rf artifacts/* logs/* data/processed/* data/raw/*
