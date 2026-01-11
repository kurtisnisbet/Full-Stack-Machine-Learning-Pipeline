from __future__ import annotations
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

from src.utils.config import load_config


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def open_image(path: Path, fallback_text: str) -> Image.Image:
    if path.exists():
        return Image.open(path).convert("RGB")
    # Create a placeholder if a figure is missing
    img = Image.new("RGB", (1200, 800), "white")
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 1199, 799], outline="black", width=3)
    d.text((40, 40), f"Missing: {fallback_text}\n{path}", fill="black")
    return img


def main(config_path: str = "config.yaml") -> None:
    cfg = load_config(config_path)
    paths = cfg["paths"]

    figures_dir = Path(paths["figures_dir"])
    ensure_dir(figures_dir)

    # Expected figures produced by make_figures.py and earlier steps
    pipeline = figures_dir / "rain_predictor_pipeline_with_config.png"
    class_dist = figures_dir / "class_distribution_train.png"
    hyper = figures_dir / "hyperparameter_curve_logreg.png"
    roc = figures_dir / "roc_curve_test.png"
    pr = figures_dir / "pr_curve_test.png"
    cm = figures_dir / "confusion_matrix.png"

    # Load images or placeholders if any are missing
    img_pipeline = open_image(pipeline, "Pipeline Diagram")
    img_class = open_image(class_dist, "Class Distribution")
    img_hyper = open_image(hyper, "Hyperparameter Curve")
    img_roc = open_image(roc, "ROC Curve (Test)")
    img_pr = open_image(pr, "PR Curve (Test)")
    img_cm = open_image(cm, "Confusion Matrix (Test)")

    # Create a large canvas (
    W, H = 2480, 3508  # ~A4 at ~300 DPI
    canvas = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(canvas)

    # Fonts
    try:
        title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 80)
        section_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 48)
    except:
        title_font = ImageFont.load_default()
        section_font = ImageFont.load_default()

    # Title
    title = "Rain Predictor — Model Results Dashboard"
    draw.text((W//2 - 700, 30), title, fill="black", font=title_font)

    # Layout grid
    margin = 60
    gap = 40

    # Top: Pipeline (full width)
    top_y = 140
    top_h = 800
    img_pipeline = img_pipeline.resize((W - 2*margin, top_h))
    canvas.paste(img_pipeline, (margin, top_y))

    # Middle row: Class Dist | Hyperparameter
    mid_y = top_y + top_h + gap
    mid_h = 800
    col_w = (W - 2*margin - gap) // 2

    img_class = img_class.resize((col_w, mid_h))
    img_hyper = img_hyper.resize((col_w, mid_h))

    canvas.paste(img_class, (margin, mid_y))
    canvas.paste(img_hyper, (margin + col_w + gap, mid_y))

    # Bottom row: ROC | PR | Confusion
    bot_y = mid_y + mid_h + gap
    bot_h = 900
    col3_w = (W - 2*margin - 2*gap) // 3

    img_roc = img_roc.resize((col3_w, bot_h))
    img_pr = img_pr.resize((col3_w, bot_h))
    img_cm = img_cm.resize((col3_w, bot_h))

    canvas.paste(img_roc, (margin, bot_y))
    canvas.paste(img_pr, (margin + col3_w + gap, bot_y))
    canvas.paste(img_cm, (margin + 2*(col3_w + gap), bot_y))

    # Footer
    footer = "Config-driven • Time-aware validation • Hyperparameter tuning • Reproducible artefacts"
    draw.text((W//2 - 900, H - 90), footer, fill="black", font=section_font)

    # Save
    out_path = figures_dir / "results_dashboard.png"
    canvas.save(out_path, "PNG")
    print(f"Dashboard saved to: {out_path}")


if __name__ == "__main__":
    main()
