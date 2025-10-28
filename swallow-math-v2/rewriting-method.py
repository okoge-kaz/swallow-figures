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

# ---- Experiment 2 (extract) ----
exp2 = [
    row("実験2 (extract)", 4.19, 0.5845, 0.3398, 0.6382, 0.2220),
    row("実験2 (extract)", 8.39, 0.5974, 0.3619, 0.6216, 0.2382),
    row("実験2 (extract)", 10.49, 0.5982, 0.3677, 0.6288, 0.2314),
    row("実験2 (extract)", 12.58, 0.6118, 0.3724, 0.6282, 0.2474),
    row("実験2 (extract)", 16.78, 0.6277, 0.3819, 0.6414, 0.2558),
    row("実験2 (extract)", 20.97, 0.6376, 0.4143, 0.6469, 0.2586),
    row("実験2 (extract)", 25.17, 0.6368, 0.4065, 0.6472, 0.2642),
    row("実験2 (extract)", 29.36, 0.6353, 0.4169, 0.6534, 0.2680),
    row("実験2 (extract)", 31.46, 0.6247, 0.4210, 0.6537, 0.2730),
    row("実験2 (extract)", 33.55, 0.6368, 0.4063, 0.6607, 0.2758),
    row("実験2 (extract)", 37.75, 0.6406, 0.4184, 0.6598, 0.2718),
    row("実験2 (extract)", 41.94, 0.6399, 0.4318, 0.6555, 0.2830),
    row("実験2 (extract)", 46.14, 0.6437, 0.4313, 0.6640, 0.2834),
    row("実験2 (extract)", 50.33, 0.6550, 0.4459, 0.6742, 0.2840),
    row("実験2 (extract)", 52.43, 0.6550, 0.4425, 0.6704, 0.2802),
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

# ---- Experiment 6 (planning) ----
exp6 = [
    row("実験6 (planning)", 4.19, 0.5572, 0.3324, 0.6589, 0.2206),
    row("実験6 (planning)", 8.39, 0.5823, 0.3490, 0.6474, 0.2128),
    row("実験6 (planning)", 10.49, 0.5807, 0.3682, 0.6636, 0.2154),
    row("実験6 (planning)", 12.58, 0.5807, 0.3620, 0.6647, 0.2424),
    row("実験6 (planning)", 16.78, 0.5921, 0.3807, 0.6567, 0.2464),
    row("実験6 (planning)", 20.97, 0.6133, 0.3892, 0.6747, 0.2432),
    row("実験6 (planning)", 25.17, 0.6103, 0.4050, 0.6604, 0.2564),
    row("実験6 (planning)", 29.36, 0.6262, 0.4067, 0.6719, 0.2462),
    row("実験6 (planning)", 31.46, 0.6202, 0.4036, 0.6698, 0.2524),
    row("実験6 (planning)", 33.55, 0.6406, 0.4115, 0.6669, 0.2624),
    row("実験6 (planning)", 37.75, 0.6300, 0.4152, 0.6730, 0.2634),
    row("実験6 (planning)", 41.94, 0.6315, 0.4246, 0.6821, 0.2608),
    row("実験6 (planning)", 46.14, 0.6247, 0.4245, 0.6845, 0.2494),
    row("実験6 (planning)", 50.33, 0.6270, 0.4376, 0.6888, 0.2568),
    row("実験6 (planning)", 52.43, 0.6194, 0.4273, 0.6928, 0.2576),
]

# ---- Experiment 7 (socratic) ----
exp7 = [
    row("実験7 (socratic)", 4.19, 0.5838, 0.3543, 0.6448, 0.2194),
    row("実験7 (socratic)", 8.39, 0.5739, 0.3524, 0.6415, 0.2244),
    row("実験7 (socratic)", 10.49, 0.5929, 0.3759, 0.6577, 0.2454),
    row("実験7 (socratic)", 12.58, 0.5997, 0.3744, 0.6514, 0.2388),
    row("実験7 (socratic)", 16.78, 0.6224, 0.3948, 0.6627, 0.2452),
    row("実験7 (socratic)", 20.97, 0.6232, 0.4000, 0.6835, 0.2444),
    row("実験7 (socratic)", 25.17, 0.6073, 0.3943, 0.6767, 0.2450),
    row("実験7 (socratic)", 29.36, 0.6535, 0.4152, 0.6899, 0.2566),
    row("実験7 (socratic)", 31.46, 0.6437, 0.4160, 0.6781, 0.2698),
    row("実験7 (socratic)", 33.55, 0.6505, 0.4167, 0.6901, 0.2652),
    row("実験7 (socratic)", 37.75, 0.6573, 0.4204, 0.6881, 0.2648),
    row("実験7 (socratic)", 41.94, 0.6596, 0.4158, 0.6848, 0.2774),
    row("実験7 (socratic)", 46.14, 0.6528, 0.4199, 0.6890, 0.2658),
    row("実験7 (socratic)", 50.33, 0.6596, 0.4273, 0.6893, 0.2724),
    row("実験7 (socratic)", 52.43, 0.6603, 0.4252, 0.6928, 0.2660),
]

# ---- Experiment 8 (multiple solution) ----
exp8 = [
    row("実験8 (multiple solution)", 4.19, 0.5823, 0.3478, 0.6546, 0.2290),
    row("実験8 (multiple solution)", 8.39, 0.5656, 0.3439, 0.6395, 0.2374),
    row("実験8 (multiple solution)", 10.49, 0.5694, 0.3630, 0.6515, 0.2416),
    row("実験8 (multiple solution)", 12.58, 0.5861, 0.3746, 0.6478, 0.2590),
    row("実験8 (multiple solution)", 16.78, 0.6133, 0.3841, 0.6452, 0.2628),
    row("実験8 (multiple solution)", 20.97, 0.6088, 0.3971, 0.6696, 0.2676),
    row("実験8 (multiple solution)", 25.17, 0.6103, 0.4000, 0.6544, 0.2724),
    row("実験8 (multiple solution)", 29.36, 0.6444, 0.4189, 0.6590, 0.2804),
    row("実験8 (multiple solution)", 31.46, 0.6285, 0.4074, 0.6719, 0.2808),
    row("実験8 (multiple solution)", 33.55, 0.6300, 0.4074, 0.6672, 0.2850),
    row("実験8 (multiple solution)", 37.75, 0.6232, 0.4139, 0.6782, 0.2898),
    row("実験8 (multiple solution)", 41.94, 0.6368, 0.4225, 0.6667, 0.2866),
    row("実験8 (multiple solution)", 46.14, 0.6422, 0.4238, 0.6778, 0.2844),
    row("実験8 (multiple solution)", 50.33, 0.6437, 0.4302, 0.6819, 0.2858),
    row("実験8 (multiple solution)", 52.43, 0.6376, 0.4339, 0.6816, 0.2830),
]

rows = [baseline] + exp1 + exp2 + exp4 + exp5 + exp6 + exp7 + exp8
df = pd.DataFrame(rows).sort_values(by=["series", "tokens_B"]).reset_index(drop=True)

# Optional ASCII aliases to avoid font warnings in legends
alias = {
    "実験1": "finemath-3+",
    "実験2 (extract)": "extract",
    "実験4 (textbook)": "textbook",
    "実験5 (QA)": "QA",
    "実験6 (planning)": "planning",
    "実験7 (socratic)": "socratic",
    "実験8 (multiple solution)": "multiple solution",
    "Meta-Llama-3.1-8B (0B)": "Meta-Llama-3.1-8B (0B)",
}


def legend_name(name: str) -> str:
    if USE_ASCII_LEGEND:
        return alias.get(name, name)
    return name


# Inject a 0B baseline point for *each experiment* series so their curves start from the same 0B value
experiment_prefixes = ("実験1", "実験2", "実験4", "実験5", "実験6", "実験7", "実験8")
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

composite_path = Path("swallow-math-v2/figures/rewriting-method.png")
composite.save(composite_path)

# Display the composite in the notebook
plt.figure()
plt.imshow(composite)
plt.axis("off")
