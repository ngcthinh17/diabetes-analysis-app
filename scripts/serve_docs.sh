#!/usr/bin/env bash
source .venv/bin/activate
pip install mkdocs mkdocs-material
mkdocs serve -a 127.0.0.1:8000
