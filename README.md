# Diabetes Analysis App (Course Project Scaffold)

A clean, organized **Streamlit** project structure for your course topics:

## Topics (choose one)
1. Re-implement your previous Train C project (plug into this scaffold).
2. Survey a dataset and build an interactive app (Streamlit).
   - THPTQG 2025 exam scores (upload your CSV)
   - Kaggle Diabetes (Pima) - upload `diabetes.csv`
   - Palmer Penguins - upload CSV
3. Your own custom topic/dataset.

## Structure
```
diabetes-analysis-app/
  app/
    __init__.py
    main.py
    pages/
      __init__.py
      data_exploration.py
      visualization.py
      prediction.py
      about.py
    utils/
      __init__.py
      data_loader.py
      ml_model.py
      visualization.py
    assets/
      style.css
  tests/
    __init__.py
    test_data_loader.py
    test_ml_model.py
  docs/
    index.md
    setup.md
    usage.md
    report.pdf
  requirements.txt
  setup.py
  .gitignore
  README.md
```

## Quickstart
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/main.py
```

## Presenting
- Keep repo ready for live demo
- Slides: problem, data, methods, results, contribution, Q&A
- Document contributions per member in README or docs/

---

## Documentation (meets course requirements)

- **Purpose of project**: See `docs/index.md`
- **Project structure**: See tree in this README
- **Setup & Run**: `docs/setup.md` and `scripts/` helpers
- **Demo**: `docs/demo.md` (capture steps)
- **Member contributions**: `docs/contributions.md`
- **Report**: `docs/report.tex` (LaTeX) and `docs/report.typ` (Typst) + a placeholder `docs/report.pdf`
- **Docs site** (MkDocs): run `scripts/serve_docs.bat` (or `.sh`) then open http://127.0.0.1:8000

### Slides
- Markdown slides at `slides/slides.md` (compatible with Marp/reveal-md)

### Testing
```bash
pytest -q
```

### Python version
- Requires **Python >= 3.9**
