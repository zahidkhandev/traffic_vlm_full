python -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 13) else f'Python 3.13 required, found {sys.version}')"
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
