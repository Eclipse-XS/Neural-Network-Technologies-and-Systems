"""Generate publication-grade scientific evaluation figures from frozen benchmark artifacts."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Set publication style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.labelweight": "bold",
    "figure.titlesize": 14,
    "figure.titleweight": "bold",
    "figure.dpi": 200,
})

REPO_ROOT = Path(__file__).resolve().parents[3]
LAB_ROOT = REPO_ROOT / "labs/lab07_final_project"
BASELINES_DIR = LAB_ROOT / "evaluation/baselines"
OUT_DIR = LAB_ROOT / "artifacts/portfolio"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load benchmark data
bench_path = BASELINES_DIR / "benchmark_results.json"
with open(bench_path, encoding="utf-8") as f:
    bench_data = json.load(f)

modes = bench_data["modes"]
dense = modes["dense"]
lexical = modes["lexical"]
hybrid = modes["hybrid"]

# Colors
COLOR_DENSE = "#2b5c8f"    # Steel Blue
COLOR_BM25 = "#e26d5c"     # Terracotta / Coral
COLOR_HYBRID = "#2a9d8f"   # Persian Green / Teal
COLOR_TARGET = "#d90429"   # Crimson Target line

# =========================================================================
# Figure 1: Recall@k Comparison (Dense vs BM25 vs Hybrid)
# =========================================================================
def plot_recall_comparison():
    fig, ax = plt.subplots(figsize=(8, 5))
    k_labels = ["Recall@1", "Recall@3", "Recall@5"]
    x = np.arange(len(k_labels))
    width = 0.25

    dense_vals = [dense["evidence_recall_1"], dense["evidence_recall_3"], dense["evidence_recall_5"]]
    bm25_vals = [lexical["evidence_recall_1"], lexical["evidence_recall_3"], lexical["evidence_recall_5"]]
    hybrid_vals = [hybrid["evidence_recall_1"], hybrid["evidence_recall_3"], hybrid["evidence_recall_5"]]

    rects1 = ax.bar(x - width, dense_vals, width, label="Dense (multilingual-e5-small)", color=COLOR_DENSE)
    rects2 = ax.bar(x, bm25_vals, width, label="Lexical (BM25)", color=COLOR_BM25)
    rects3 = ax.bar(x + width, hybrid_vals, width, label="Hybrid (RRF, k=60)", color=COLOR_HYBRID)

    # Threshold targets
    targets = [0.65, 0.80, 0.85]
    for i, t in enumerate(targets):
        ax.plot([i - 0.4, i + 0.4], [t, t], color=COLOR_TARGET, linestyle="--", linewidth=1.5,
                label="Target Threshold" if i == 0 else "")

    ax.set_ylabel("Evidence Recall Score")
    ax.set_title("Evidence Recall@K Across Retrieval Modes\n(100-Query Frozen Scientific Benchmark, n=85 answerable)")
    ax.set_xticks(x)
    ax.set_xticklabels(k_labels)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", frameon=True, framealpha=0.9)

    # Value labels
    def add_labels(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.3f}",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8.5, weight="bold")

    add_labels(rects1)
    add_labels(rects2)
    add_labels(rects3)

    plt.tight_layout()
    out_path = OUT_DIR / "retrieval_recall_comparison.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Generated: {out_path}")

# =========================================================================
# Figure 2: MRR & Hit@5 Comparison
# =========================================================================
def plot_mrr_hit_comparison():
    fig, ax = plt.subplots(figsize=(7, 5))
    metrics = ["Mean Reciprocal Rank (MRR)", "Hit@5"]
    x = np.arange(len(metrics))
    width = 0.25

    dense_vals = [dense["mrr"], dense["hit_at_5"]]
    bm25_vals = [lexical["mrr"], lexical["hit_at_5"]]
    hybrid_vals = [hybrid["mrr"], hybrid["hit_at_5"]]

    rects1 = ax.bar(x - width, dense_vals, width, label="Dense Only", color=COLOR_DENSE)
    rects2 = ax.bar(x, bm25_vals, width, label="Lexical (BM25)", color=COLOR_BM25)
    rects3 = ax.bar(x + width, hybrid_vals, width, label="Hybrid RRF", color=COLOR_HYBRID)

    # Targets: MRR >= 0.75, Hit@5 >= 0.92
    targets = [0.75, 0.92]
    for i, t in enumerate(targets):
        ax.plot([i - 0.4, i + 0.4], [t, t], color=COLOR_TARGET, linestyle="--", linewidth=1.5,
                label="Quality Target" if i == 0 else "")

    ax.set_ylabel("Metric Value")
    ax.set_title("MRR and Hit@5 Retrieval Quality\n(100-Query Frozen Benchmark)")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", frameon=True, framealpha=0.9)

    for rects in [rects1, rects2, rects3]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.3f}",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9, weight="bold")

    plt.tight_layout()
    out_path = OUT_DIR / "retrieval_mrr_hit_comparison.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Generated: {out_path}")

# =========================================================================
# Figure 3: Latency Profile Comparison
# =========================================================================
def plot_latency_profile():
    fig, ax = plt.subplots(figsize=(8, 4.5))
    modes_list = ["Lexical (BM25)", "Dense (CPU)", "Hybrid (Dense+BM25)"]
    x = np.arange(len(modes_list))
    width = 0.2

    p50 = [lexical["latency"]["p50_ms"], dense["latency"]["p50_ms"], hybrid["latency"]["p50_ms"]]
    p90 = [lexical["latency"]["p90_ms"], dense["latency"]["p90_ms"], hybrid["latency"]["p90_ms"]]
    p95 = [lexical["latency"]["p95_ms"], dense["latency"]["p95_ms"], hybrid["latency"]["p95_ms"]]
    mean_lat = [lexical["latency"]["mean_ms"], dense["latency"]["mean_ms"], hybrid["latency"]["mean_ms"]]

    r1 = ax.bar(x - 1.5 * width, p50, width, label="P50 (Median)", color="#457b9d")
    r2 = ax.bar(x - 0.5 * width, mean_lat, width, label="Mean Latency", color="#2a9d8f")
    r3 = ax.bar(x + 0.5 * width, p90, width, label="P90", color="#e76f51")
    r4 = ax.bar(x + 1.5 * width, p95, width, label="P95", color="#e63946")

    ax.set_ylabel("Latency (milliseconds)")
    ax.set_title("Retrieval Latency Distribution Across Modes\n(Measured on Local Intel/AMD CPU Runtime, 100 Queries)")
    ax.set_xticks(x)
    ax.set_xticklabels(modes_list)
    ax.set_ylim(0, max(p95) * 1.25)
    ax.legend(loc="upper left", frameon=True)

    for rects in [r1, r2, r3, r4]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.1f}ms",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 2), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7.5)

    plt.tight_layout()
    out_path = OUT_DIR / "retrieval_latency_distribution.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Generated: {out_path}")

# =========================================================================
# Figure 4: Category Breakdown Analysis (Hybrid Mode)
# =========================================================================
def plot_category_breakdown():
    fig, ax = plt.subplots(figsize=(9, 5))
    cats = hybrid["category_breakdown"]
    cat_names = [
        "Exact Fact\n(n=25)",
        "Semantic Paraphrase\n(n=20)",
        "Cross-Document\n(n=20)",
        "Numerical Reasoning\n(n=20)",
    ]
    keys = ["exact_fact", "semantic_paraphrase", "cross_document", "numerical_reasoning"]
    x = np.arange(len(cat_names))
    width = 0.22

    hit5 = [cats[k]["hit_5"] for k in keys]
    mrr = [cats[k]["mrr"] for k in keys]
    r1 = [cats[k]["recall_1"] for k in keys]
    r5 = [cats[k]["recall_5"] for k in keys]

    rects1 = ax.bar(x - 1.5 * width, r1, width, label="Recall@1", color="#3d5a80")
    rects2 = ax.bar(x - 0.5 * width, r5, width, label="Recall@5", color="#98c1d9")
    rects3 = ax.bar(x + 0.5 * width, hit5, width, label="Hit@5", color="#ee6c4d")
    rects4 = ax.bar(x + 1.5 * width, mrr, width, label="MRR", color="#293241")

    ax.set_ylabel("Metric Value")
    ax.set_title("Hybrid Retrieval Performance by Query Complexity Category\n(100-Query Frozen Benchmark, 15 Negative Controls Refused 100%)")
    ax.set_xticks(x)
    ax.set_xticklabels(cat_names)
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper right", frameon=True, ncol=2)

    for rects in [rects1, rects2, rects3, rects4]:
        for rect in rects:
            h = rect.get_height()
            ax.annotate(f"{h:.2f}",
                        xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 2), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    out_path = OUT_DIR / "category_performance_breakdown.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Generated: {out_path}")

if __name__ == "__main__":
    plot_recall_comparison()
    plot_mrr_hit_comparison()
    plot_latency_profile()
    plot_category_breakdown()
    print("All scientific visualization plots generated successfully!")
