@echo off
call .venv\Scripts\activate
pip install mkdocs mkdocs-material
mkdocs build
