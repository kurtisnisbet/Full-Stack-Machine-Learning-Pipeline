ASSET_DIR = asset

.PHONY: all ingest clean features train evaluate figures dashboard run test lint app help

# ── Default: run the full pipeline ───────────────────────────────────────────
all: ingest clean features train evaluate figures dashboard

# ── Individual pipeline stages ───────────────────────────────────────────────
ingest:
	cd $(ASSET_DIR) && python -m src.data.ingest

clean:
	cd $(ASSET_DIR) && python -m src.data.clean

features:
	cd $(ASSET_DIR) && python -m src.features.build_features

train:
	cd $(ASSET_DIR) && python -m src.models.train

evaluate:
	cd $(ASSET_DIR) && python -m src.models.evaluate

figures:
	cd $(ASSET_DIR) && python -m src.reports.make_figures

dashboard:
	cd $(ASSET_DIR) && python -m src.reports.make_dashboard

# ── Tests & linting ──────────────────────────────────────────────────────────
test:
	pytest tests/ -v

lint:
	flake8 $(ASSET_DIR)/src/ --max-line-length=120 --extend-ignore=E501,W503

# ── Streamlit app ─────────────────────────────────────────────────────────────
app:
	streamlit run app.py

# ── Help ─────────────────────────────────────────────────────────────────────
help:
	@echo "Available targets:"
	@echo "  make all        Run the full pipeline end-to-end"
	@echo "  make ingest     CSV -> interim Parquet + audit reports"
	@echo "  make clean      Interim -> cleaned dataset"
	@echo "  make features   Cleaned -> train/val/test splits + encoding"
	@echo "  make train      Train all models via hyperparameter grid search"
	@echo "  make evaluate   Evaluate best model; optimise decision threshold"
	@echo "  make figures    Generate diagnostic plots (ROC, PR, SHAP, ...)"
	@echo "  make dashboard  Compile figures into a single results dashboard"
	@echo "  make test       Run pytest unit tests"
	@echo "  make lint       Run flake8 linter"
	@echo "  make app        Launch interactive Streamlit prediction app"
