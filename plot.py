import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import Counter

# ── global font sizes ──
TITLE_SIZE    = 30
LABEL_SIZE    = 25
TICK_SIZE     = 25
BAR_TEXT_SIZE = 25
LEGEND_SIZE   = 20
AXIS_SIZE     = 25

# ── spacing config ──
PAIR_GAP   = 1.0   # space between baseline and paraphrased within a pair
GROUP_GAP  = 1.5   # space between different pairs/groups
BAR_HEIGHT = 0.6

def get_y_positions(n_hate_bars, extra_count=0):
    positions = []
    y = 0
    for i in range(n_hate_bars):
        positions.append(y)
        if i % 2 == 0:
            y += PAIR_GAP
        else:
            y += GROUP_GAP
    # extra conditions — no extra gap added since GROUP_GAP was already added after last pair
    for _ in range(extra_count):
        positions.append(y)
        y += GROUP_GAP
    return positions

def load_jsonl_files(filepaths):
    """Load and aggregate multiple jsonl files into a list of entries."""
    entries = []
    for filepath in filepaths:
        with open(filepath, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        print(f"[WARNING] Could not parse line in {filepath}")
    return entries

def get_score_distribution(entries):
    """Get normalized score distribution from entries."""
    counts = Counter(e["score"] for e in entries)
    total = len(entries)
    return {
        "counts": [counts.get(0, 0), counts.get(1, 0), counts.get(2, 0)],
        "total": total
    }

def build_conditions(data, mode):
    conditions = {}

    if mode == "aggregate":
        all_baseline    = []
        all_paraphrased = []
        for category, requests in data.items():
            for request_id, files in requests.items():
                all_baseline    += files["baseline"]
                all_paraphrased += files["paraphrased"]
        conditions["Hate\nBaseline"]    = all_baseline
        conditions["Hate\nParaphrased"] = all_paraphrased

    elif mode == "category":
        for category, requests in data.items():
            cat_baseline    = []
            cat_paraphrased = []
            for request_id, files in requests.items():
                cat_baseline    += files["baseline"]
                cat_paraphrased += files["paraphrased"]
            conditions[f"{category}\nBaseline"]    = cat_baseline
            conditions[f"{category}\nParaphrased"] = cat_paraphrased

    elif mode == "request":
        for category, requests in data.items():
            for request_id, files in requests.items():
                conditions[f"{category} #{request_id}\nBaseline"]    = files["baseline"]
                conditions[f"{category} #{request_id}\nParaphrased"] = files["paraphrased"]

    return conditions

def plot_model_figure(model_name, data, mode="aggregate", extra_conditions=None, output_path=None):
    colors = {0: "#d32f2f", 1: "#f9a825", 2: "#388e3c"}

    conditions = build_conditions(data, mode)
    n_hate_bars = len(conditions)

    if extra_conditions:
        conditions.update(extra_conditions)

    labels = list(conditions.keys())
    extra_count = len(extra_conditions) if extra_conditions else 0
    y_positions = get_y_positions(n_hate_bars, extra_count)

    fig, ax = plt.subplots(figsize=(14, max(y_positions) * 1.2 + 2))

    for i, (label, filepaths) in enumerate(conditions.items()):
        y = y_positions[i]

        # ── pair shading ──
        if i < n_hate_bars:
            pair_index = i // 2
            if pair_index % 2 == 0:
                ax.axhspan(y - BAR_HEIGHT, y + BAR_HEIGHT, color="lightgray", alpha=0.15, zorder=0)
            if i % 2 == 0 and i > 0:
                ax.axhline(y=y - GROUP_GAP/2, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)

        # ── solid divider before extra conditions ──
        if i == n_hate_bars:
            ax.axhline(y=y - GROUP_GAP/2, color="gray", linestyle="-", linewidth=1.2)

        entries = load_jsonl_files(filepaths)
        dist = get_score_distribution(entries)
        total = dist["total"]
        counts = dist["counts"]

        left = 0
        for score, count in enumerate(counts):
            pct = count / total * 100 if total > 0 else 0
            ax.barh(y, pct, left=left, color=colors[score], edgecolor="white", linewidth=0.5, height=BAR_HEIGHT)
            if count > 0:
                ax.text(left + pct / 2, y, str(count),
                        ha="center", va="center",
                        fontsize=BAR_TEXT_SIZE, color="white", fontweight="bold")
            left += pct

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=TICK_SIZE)
    ax.set_xlabel("Percentage (%)", fontsize=AXIS_SIZE)
    ax.set_xlim(0, 100)
    ax.set_title(f"Score Distribution — {model_name} ({mode})",
                 fontsize=TITLE_SIZE, fontweight="bold", pad=15)
    ax.tick_params(axis='x', labelsize=TICK_SIZE)

    legend_elements = [
        plt.Rectangle((0,0), 1, 1, color=colors[0], label="0"),
        plt.Rectangle((0,0), 1, 1, color=colors[1], label="1"),
        plt.Rectangle((0,0), 1, 1, color=colors[2], label="2"),
    ]
    ax.legend(handles=legend_elements, title="Score", loc="upper right",
              fontsize=LEGEND_SIZE, title_fontsize=LEGEND_SIZE + 1)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {output_path}")
    else:
        plt.show()

def plot_comparison_figure(models, output_path=None):
    """
    Plot hate vs other baseline for multiple models side by side in one figure.
    
    models: list of dicts:
    [
        {
            "name": "Gemini-2.5-flash",
            "hate":  [filepath, ...],
            "other": [filepath, ...]
        },
        ...
    ]
    """
    colors = {0: "#d32f2f", 1: "#f9a825", 2: "#388e3c"}

    # build conditions: for each model, hate then other
    all_labels     = []
    all_filepaths  = []
    all_y          = []
    divider_ys     = []

    y = 0
    for m, model in enumerate(models):
        # divider between models
        if m > 0:
            divider_ys.append(y - GROUP_GAP / 2)

        for condition in ["hate", "other"]:
            all_labels.append(f"{model['name']}\n{condition.capitalize()} Baseline")
            all_filepaths.append(model[condition])
            all_y.append(y)
            y += PAIR_GAP

        y += GROUP_GAP - PAIR_GAP  # extra gap between models

    fig, ax = plt.subplots(figsize=(14, max(all_y) * 1.4 + 2))

    for i, (label, filepaths) in enumerate(zip(all_labels, all_filepaths)):
        y = all_y[i]

        # shading per model block
        model_index = i // 2
        if model_index % 2 == 0:
            ax.axhspan(y - BAR_HEIGHT, y + BAR_HEIGHT, color="lightgray", alpha=0.15, zorder=0)

        entries = load_jsonl_files(filepaths)
        dist = get_score_distribution(entries)
        total = dist["total"]
        counts = dist["counts"]

        left = 0
        for score, count in enumerate(counts):
            pct = count / total * 100 if total > 0 else 0
            ax.barh(y, pct, left=left, color=colors[score],
                    edgecolor="white", linewidth=0.5, height=BAR_HEIGHT)
            if count > 0:
                ax.text(left + pct / 2, y, str(count),
                        ha="center", va="center",
                        fontsize=BAR_TEXT_SIZE, color="white", fontweight="bold")
            left += pct

    # dividers between models
    for dy in divider_ys:
        ax.axhline(y=dy, color="gray", linestyle="-", linewidth=1.2)

    ax.set_yticks(all_y)
    ax.set_yticklabels(all_labels, fontsize=TICK_SIZE)
    ax.set_xlabel("Percentage (%)", fontsize=AXIS_SIZE)
    ax.set_xlim(0, 100)
    ax.set_title("Score Distribution — Google Models",
                 fontsize=TITLE_SIZE, fontweight="bold", pad=15)
    ax.tick_params(axis='x', labelsize=TICK_SIZE)

    legend_elements = [
        plt.Rectangle((0,0), 1, 1, color=colors[0], label="0"),
        plt.Rectangle((0,0), 1, 1, color=colors[1], label="1"),
        plt.Rectangle((0,0), 1, 1, color=colors[2], label="2"),
    ]
    ax.legend(handles=legend_elements, title="Score", loc="lower right",
              fontsize=LEGEND_SIZE, title_fontsize=LEGEND_SIZE + 1)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {output_path}")
    else:
        plt.show()

# ── CONFIGURE FILES ──────────────────────────────────────────────────

base_dir = Path(__file__).parent

gpt4o_data = {
    "Racial": {
        "100": {
            "baseline":    [base_dir / "repeated_results_openai_gpt-4o-mini_hate_100.jsonl"],
            "paraphrased": [base_dir / "repeated_results_openai_gpt-4o-mini_paraphrased_0.jsonl"],
        },
        "101": {
            "baseline":    [base_dir / "repeated_results_openai_gpt-4o-mini_hate_101.jsonl"],
            "paraphrased": [base_dir / "repeated_results_openai_gpt-4o-mini_paraphrased_1.jsonl"],
        },
    },
    "Religious": {
        "110": {
            "baseline":    [base_dir / "repeated_results_openai_gpt-4o-mini_hate_110.jsonl"],
            "paraphrased": [base_dir / "repeated_results_openai_gpt-4o-mini_paraphrased_10.jsonl"],
        },
        "115": {
            "baseline":    [base_dir / "repeated_results_openai_gpt-4o-mini_hate_115.jsonl"],
            "paraphrased": [base_dir / "repeated_results_openai_gpt-4o-mini_paraphrased_15.jsonl"],
        },
    },
    "Gender": {
        "123": {
            "baseline":    [base_dir / "repeated_results_openai_gpt-4o-mini_hate_123.jsonl"],
            "paraphrased": [base_dir / "repeated_results_openai_gpt-4o-mini_paraphrased_23.jsonl"],
        },
        "126": {
            "baseline":    [base_dir / "repeated_results_openai_gpt-4o-mini_hate_126.jsonl"],
            "paraphrased": [base_dir / "repeated_results_openai_gpt-4o-mini_paraphrased_26.jsonl"],
        },
    },
}

gpt4o_extra = {
    "Other\nBaseline": [base_dir / "repeated_results_openai_gpt-4o-mini_other.jsonl"],
}

# for mode in ["aggregate", "category", "request"]:
#     plot_model_figure(
#         model_name="GPT-4o-mini",
#         data=gpt4o_data,
#         mode=mode,
#         extra_conditions=gpt4o_extra,
#         output_path=base_dir / f"figure_gpt4o_mini_{mode}.png"
#     )

# ── GOOGLE MODELS ──────────────────────────────────────────────────────────────

plot_comparison_figure(
    models=[
        {
            "name":  "Gemini-2.5-flash",
            "hate":  [base_dir / "sampled_results_gemini-2.5-flash.jsonl"],
            "other": [base_dir / "repeated_results_google_gemini-2.5-flash_other.jsonl"],
        },
        {
            "name":  "Gemma-4-31b-it",
            "hate":  [base_dir / "sampled_results_gemma-4-31b-it.jsonl"],
            "other": [base_dir / "repeated_results_gemma-4-31b-it_other.jsonl"],
        },
    ],
    output_path=base_dir / "figure_google_models.png"
)