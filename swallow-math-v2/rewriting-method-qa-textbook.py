import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from PIL import Image

USE_ASCII_LEGEND = True

# Baseline (0B tokens)
baseline = {
    "series": "Meta-Llama-3.1-8B (0B)",
    "tokens_B": 0.0,
    "GSM8K": 0.5679,
    "GSM-Plus": 0.3145,
    "BBH": 0.6292,
    "MATH": 0.1816,
}


def row(series, tokens, g8k, gplus, bbh, math_):
    return {
        "series": series,
        "tokens_B": tokens,
        "GSM8K": g8k,
        "GSM-Plus": gplus,
        "BBH": bbh,
        "MATH": math_,
    }


# ---- Experiment 1
exp1 = [
    row("実験1", 4.19, 0.5830, 0.3232, 0.6173, 0.1956),
    row("実験1", 8.39, 0.5982, 0.3106, 0.6096, 0.1974),
    row("実験1", 10.49, 0.5982, 0.3186, 0.6205, 0.2012),
    row("実験1", 12.58, 0.6126, 0.3345, 0.6143, 0.1984),
    row("実験1", 16.78, 0.6293, 0.3380, 0.6099, 0.1970),
    row("実験1", 20.97, 0.6376, 0.3555, 0.6312, 0.1974),
    row("実験1", 25.17, 0.6368, 0.3570, 0.6080, 0.2174),
    row("実験1", 29.36, 0.6353, 0.3740, 0.6279, 0.2142),
    row("実験1", 31.46, 0.6240, 0.3786, 0.6208, 0.2106),
    row("実験1", 33.55, 0.6391, 0.3741, 0.6335, 0.2222),
    row("実験1", 37.75, 0.6414, 0.3641, 0.6358, 0.2228),
    row("実験1", 41.94, 0.6422, 0.3723, 0.6314, 0.2204),
    row("実験1", 46.14, 0.6452, 0.3813, 0.6429, 0.2300),
    row("実験1", 50.33, 0.6550, 0.3826, 0.6428, 0.2288),
    row("実験1", 52.43, 0.6480, 0.3785, 0.6406, 0.2314),
]

# ---- Experiment 4 (textbook) ----
exp4 = [
    row("実験4 (textbook)", 4.19, 0.5898, 0.3546, 0.6676, 0.2208),
    row("実験4 (textbook)", 8.39, 0.5656, 0.3519, 0.6518, 0.2316),
    row("実験4 (textbook)", 10.49, 0.5921, 0.3672, 0.6696, 0.2422),
    row("実験4 (textbook)", 12.58, 0.5936, 0.3803, 0.6612, 0.2580),
    row("実験4 (textbook)", 16.78, 0.6217, 0.4079, 0.6601, 0.2634),
    row("実験4 (textbook)", 20.97, 0.6300, 0.4190, 0.6790, 0.2676),
    row("実験4 (textbook)", 25.17, 0.6247, 0.4170, 0.6649, 0.2678),
    row("実験4 (textbook)", 29.36, 0.6376, 0.4293, 0.6838, 0.2816),
    row("実験4 (textbook)", 31.46, 0.6293, 0.4249, 0.6871, 0.2696),
    row("実験4 (textbook)", 33.55, 0.6429, 0.4303, 0.6835, 0.2802),
    row("実験4 (textbook)", 37.75, 0.6452, 0.4402, 0.6901, 0.2886),
    row("実験4 (textbook)", 41.94, 0.6550, 0.4523, 0.6876, 0.2908),
    row("実験4 (textbook)", 46.14, 0.6475, 0.4470, 0.6962, 0.2886),
    row("実験4 (textbook)", 50.33, 0.6573, 0.4597, 0.6901, 0.2942),
    row("実験4 (textbook)", 52.43, 0.6535, 0.4587, 0.6970, 0.3014),
]

# ---- Experiment 5 (QA) ----
exp5 = [
    row("実験5 (QA)", 4.19, 0.5944, 0.3710, 0.6494, 0.2418),
    row("実験5 (QA)", 8.39, 0.6020, 0.3872, 0.6385, 0.2370),
    row("実験5 (QA)", 10.49, 0.6080, 0.3851, 0.6397, 0.2444),
    row("実験5 (QA)", 12.58, 0.5807, 0.3881, 0.6355, 0.2566),
    row("実験5 (QA)", 16.78, 0.6202, 0.4038, 0.6460, 0.2566),
    row("実験5 (QA)", 20.97, 0.6353, 0.4233, 0.6633, 0.2634),
    row("実験5 (QA)", 25.17, 0.6588, 0.4282, 0.6523, 0.2808),
    row("実験5 (QA)", 29.36, 0.6710, 0.4526, 0.6772, 0.2812),
    row("実験5 (QA)", 31.46, 0.6596, 0.4404, 0.6848, 0.2864),
    row("実験5 (QA)", 33.55, 0.6672, 0.4445, 0.6781, 0.2812),
    row("実験5 (QA)", 37.75, 0.6649, 0.4454, 0.6851, 0.2894),
    row("実験5 (QA)", 41.94, 0.6710, 0.4526, 0.6804, 0.2856),
    row("実験5 (QA)", 46.14, 0.6687, 0.4562, 0.6789, 0.2942),
    row("実験5 (QA)", 50.33, 0.6755, 0.4643, 0.6891, 0.2926),
    row("実験5 (QA)", 52.43, 0.6831, 0.4622, 0.6887, 0.2908),
]


rows = [baseline] + exp1 + exp4 + exp5
df = pd.DataFrame(rows).sort_values(by=["series", "tokens_B"]).reset_index(drop=True)

# Optional ASCII aliases to avoid font warnings in legends
alias = {
    "実験1": "finemath-3+",
    "実験4 (textbook)": "textbook",
    "実験5 (QA)": "QA",
    "Meta-Llama-3.1-8B (0B)": "Meta-Llama-3.1-8B (0B)",
}


def legend_name(name: str) -> str:
    if USE_ASCII_LEGEND:
        return alias.get(name, name)
    return name


# Inject a 0B baseline point for *each experiment* series so their curves start from the same 0B value
experiment_prefixes = ("実験1", "実験4", "実験5")
baseline_row = df[df["series"] == "Meta-Llama-3.1-8B (0B)"].iloc[0].to_dict()

augmented_rows = []
for sname, sdf in df.groupby("series"):
    if sname.startswith(experiment_prefixes):
        zero_point = {
            "series": sname,
            "tokens_B": 0.0,
            "GSM8K": baseline_row["GSM8K"],
            "GSM-Plus": baseline_row["GSM-Plus"],
            "BBH": baseline_row["BBH"],
            "MATH": baseline_row["MATH"],
        }
        augmented_rows.append(zero_point)
    augmented_rows.extend(sdf.to_dict(orient="records"))

df_aug = (
    pd.DataFrame(augmented_rows)
    .sort_values(by=["series", "tokens_B"])
    .reset_index(drop=True)
)


# Helper to plot a single metric as one chart (no subplots), save to disk, and return path
def plot_metric(metric_name: str, df_local: pd.DataFrame, outdir: Path) -> Path:
    plt.figure()
    for series_name, sdf in df_local.groupby("series"):
        if series_name == "Meta-Llama-3.1-8B (0B)":
            continue
        sdf = sdf.sort_values("tokens_B")
        plt.plot(
            sdf["tokens_B"],
            sdf[metric_name],
            marker="o",
            label=legend_name(series_name),
        )
    plt.title(f"{metric_name}")
    plt.xlabel("Tokens (B)")
    plt.ylabel("Accuracy")
    plt.legend(loc="best", fontsize=8)
    plt.grid(True)
    out_path = outdir / f"{metric_name.replace('/', '_')}.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    return out_path


# Generate four separate charts
outdir = Path("swallow-math-v2/figures")
outdir.mkdir(parents=True, exist_ok=True)
paths = []
for metric in ["GSM8K", "GSM-Plus", "BBH", "MATH"]:
    paths.append(plot_metric(metric, df_aug, outdir))

# Stitch them into one composite image (2x2 grid) using PIL
imgs = [Image.open(p) for p in paths]
min_w = min(im.width for im in imgs)
min_h = min(im.height for im in imgs)
imgs_resized = [im.resize((min_w, min_h)) for im in imgs]

cols, rows_grid = 2, 2
composite = Image.new("RGB", (cols * min_w, rows_grid * min_h), "white")
positions = [(0, 0), (min_w, 0), (0, min_h), (min_w, min_h)]
for im, pos in zip(imgs_resized, positions):
    composite.paste(im, pos)

composite_path = Path("swallow-math-v2/figures/rewriting-method-qa-textbook.png")
composite.save(composite_path)

# Display the composite in the notebook
plt.figure()
plt.imshow(composite)
plt.axis("off")
