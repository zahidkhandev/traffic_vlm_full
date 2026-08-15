.PHONY: bootstrap run test benchmark lint typecheck docker-build podman-build

bootstrap:
	python -m venv .venv
	. .venv/bin/activate && python -m pip install --upgrade pip && pip install -r requirements.txt && pip install -e .

run:
	. .venv/bin/activate && python run_cli.py run --config configs/local.yaml

benchmark:
	. .venv/bin/activate && python run_cli.py benchmark --config configs/local.yaml

test:
	. .venv/bin/activate && pytest tests --basetemp=pytest_tmp -p no:cacheprovider

lint:
	. .venv/bin/activate && ruff check qwen_auto_qc tests && ruff format --check qwen_auto_qc tests

typecheck:
	. .venv/bin/activate && mypy qwen_auto_qc

docker-build:
	docker build -t qwen-auto-qc:local .

podman-build:
	podman build -f Containerfile -t qwen-auto-qc:local .
