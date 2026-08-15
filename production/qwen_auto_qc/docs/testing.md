# Testing

## What is covered now

Current tests cover:

- config loading
- env var expansion
- dataset box filtering
- inference prompt formatting
- inference response parsing
- scoring logic
- pipeline output persistence
- unresolved required `${ENV_VAR}` placeholders fail config validation

## Test files

- `tests/test_config.py`
- `tests/test_config_env.py`
- `tests/test_dataset.py`
- `tests/test_inference.py`
- `tests/test_pipeline.py`
- `tests/test_scoring.py`

## Run tests

```bash
pytest tests --basetemp=pytest_tmp -p no:cacheprovider --cov=qwen_auto_qc --cov-report=term-missing --cov-fail-under=70
```

## What these tests do not cover yet

- real Qwen model loading
- real image inference
- real MLflow backend round-trip
- real Docker runtime

## Next testing priorities

1. one real local smoke test with actual model
2. one MLflow local integration test
3. one Docker smoke run
4. one Podman smoke run
5. one regression test on fixed sample subset
