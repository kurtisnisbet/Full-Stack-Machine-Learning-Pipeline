"""
make_dashboard.py — Compile all diagnostic figures into a single A4 dashboard PNG.

Layout (4 rows):
    Row 1  Pipeline diagram                     (full width)
    Row 2  Class distribution | Hyperparameter curve | Model comparison
    Row 3  ROC curve | PR curve | Confusion matrix
    Row 4  SHAP summary | Calibration curve | Threshold curve

Usage:
    python -m src.reports.make_dashboard
    # or via Makefile:  make dashboard
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from src.utils.config import load_config


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def open_image(path: Path, fallback_text: str) -> Image.Image:
    """Load an image, or return a labelled placeholder if it is missing."""
    if path.exists():
        return Image.open(path).convert("RGB")
    img = Image.new("RGB", (1200, 800), color=(248, 248, 248))
    d   = ImageDraw.Draw(img)
    d.rectangle([4, 4, 1195, 795], outline=(180, 180, 180), width=3)
    d.text((40, 40), f"[Missing]\n{fallback_text}", fill=(120, 120, 120))
    return img


def resize_to(img: Image.Image, w: int, h: int) -> Image.Image:
    return img.resize((w, h), Image.LANCZOS)


def main(config_path: str = "config.yaml") -> None:
    cfg   = load_config(config_path)
    paths = cfg["paths"]

    figures_dir = Path(paths["figures_dir"])
    ensure_dir(figures_dir)

    # ── Load all figures ──────────────────────────────────────────────────────
    img_pipeline  = open_image(figures_dir / "rain_predictor_pipeline_with_config.png", "Pipeline Diagram")
    img_class     = open_image(figures_dir / "class_distribution_train.png",           "Class Distribution")
    img_hyper     = open_image(figures_dir / "hyperparameter_curve_logreg.png",        "Hyperparameter Curve")
    img_model_cmp = open_image(figures_dir / "model_comparison.png",                  "Model Comparison")
    img_roc       = open_image(figures_dir / "roc_curve_test.png",                    "ROC Curve")
    img_pr        = open_image(figures_dir / "pr_curve_test.png",                     "PR Curve")
    img_cm        = open_image(figures_dir / "confusion_matrix.png",                  "Confusion Matrix (Test)")
    img_shap      = open_image(figures_dir / "shap_summary.png",                      "SHAP Summary")
    img_cal       = open_image(figures_dir / "calibration_curve.png",                 "Calibration Curve")
    img_thresh    = open_image(figures_dir / "threshold_curve.png",                   "Threshold Curve")

    # ── Canvas ────────────────────────────────────────────────────────────────
    W, H   = 2480, 4200  # ~A3 tall at 300 DPI; keeps each cell readable
    canvas = Image.new("RGB", (W, H), "white")
    draw   = ImageDraw.Draw(canvas)

    # Fonts (fall back gracefully if DejaVu not present)
    try:
        title_font   = ImageFont.truetype("DejaVuSans-Bold.ttf",  80)
        section_font = ImageFont.truetype("DejaVuSans-Bold.ttf",  48)
        footer_font  = ImageFont.truetype("DejaVuSans.ttf",       36)
    except OSError:
        title_font = section_font = footer_font = ImageFont.load_default()

    # Title
    draw.text((W // 2 - 750, 28), "Rain Predictor — Model Results Dashboard",
              fill="black", font=title_font)

    margin = 60
    gap    = 30

    # ── Row 1: Pipeline (full width) ─────────────────────────────────────────
    r1_y = 140
    r1_h = 700
    canvas.paste(resize_to(img_pipeline, W - 2 * margin, r1_h), (margin, r1_y))

    # ── Row 2: Class dist | Hyperparameter | Model comparison ────────────────
    r2_y  = r1_y + r1_h + gap
    r2_h  = 780
    col3w = (W - 2 * margin - 2 * gap) // 3

    canvas.paste(resize_to(img_class,     col3w, r2_h), (margin,                    r2_y))
    canvas.paste(resize_to(img_hyper,     col3w, r2_h), (margin + col3w + gap,      r2_y))
    canvas.paste(resize_to(img_model_cmp, col3w, r2_h), (margin + 2 * (col3w + gap), r2_y))

    # ── Row 3: ROC | PR | Confusion matrix ───────────────────────────────────
    r3_y = r2_y + r2_h + gap
    r3_h = 820

    canvas.paste(resize_to(img_roc, col3w, r3_h), (margin,                    r3_y))
    canvas.paste(resize_to(img_pr,  col3w, r3_h), (margin + col3w + gap,      r3_y))
    canvas.paste(resize_to(img_cm,  col3w, r3_h), (margin + 2 * (col3w + gap), r3_y))

    # ── Row 4: SHAP | Calibration | Threshold ────────────────────────────────
    r4_y = r3_y + r3_h + gap
    r4_h = 820

    canvas.paste(resize_to(img_shap,   col3w, r4_h), (margin,                    r4_y))
    canvas.paste(resize_to(img_cal,    col3w, r4_h), (margin + col3w + gap,      r4_y))
    canvas.paste(resize_to(img_thresh, col3w, r4_h), (margin + 2 * (col3w + gap), r4_y))

    # ── Footer ────────────────────────────────────────────────────────────────
    footer = (
        "Config-driven  •  Time-aware validation  •  Multi-model comparison  •  "
        "SHAP explainability  •  Threshold optimisation  •  Reproducible artefacts"
    )
    draw.text((margin, H - 80), footer, fill=(100, 100, 100), font=footer_font)

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = figures_dir / "results_dashboard.png"
    canvas.save(out_path, "PNG")
    print(f"Dashboard saved to: {out_path}")
    print(f"Canvas size: {W}×{H} px")


if __name__ == "__main__":
    main()
