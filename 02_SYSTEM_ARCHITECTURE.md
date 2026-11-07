# System architecture

This project is a small Python application, not a React/MERN application. The existing notebook is the research source, while FastAPI serves static HTML/CSS/JavaScript and an exported inference model. A feature-oriented frontend or database is unnecessary at this scope. The deviation from a TypeScript frontend/backend structure avoids duplicating a Python ML stack or introducing build tooling for one page.

## Layout and boundaries

```text
main.py                         FastAPI entrypoint and static-file wiring
backend/
  routes/api.py                 HTTP routes; delegates prediction
  models/schemas.py             Pydantic request validation
  models/decision_tree.json     Versioned, non-pickle trained inference artifact
  services/predictor.py         Portable tree loading and inference
frontend/
  templates/index.html          Semantic user interface shell
  static/style.css              Responsive presentation, including generated decorative hero image
  static/app.js                 Form and benchmark API interactions
ml/
  train.py                      Dataset sampling, train/test, CV, export
  results.json                  Recorded benchmark evidence
notebooks/original-colab.ipynb  Original notebook without saved outputs
tests/test_api.py               HTTP, validation, and metric checks
requirements*.txt               Runtime, training, and test dependencies
vercel.md                       Deployment handoff
```

`main.py` depends on routes and the frontend asset paths. Routes validate input with schemas and call the inference service; they contain no training logic. The predictor reads only a trusted local JSON artifact and uses Python's standard library; it cannot retrain, access network data, or load arbitrary user-provided files. Training depends on the public parquet datasets and ML packages, and exports the artifact and results. `app.js` calls only same-origin API routes, prevents stale results after edits, and does not implement model logic. The responsive table presents labeled result cards on narrow screens. UI assets and tests do not import the training module. The service does not persist user inputs, but platform logs and other infrastructure may have separate retention.

Keep business logic in services, request parsing in schemas/routes, and frontend presentation in `frontend/`. Do not commit datasets, credentials, environment files, or untrusted pickles. Update this document when module boundaries change. The included model and validation scores are research artifacts, not production security guarantees.
