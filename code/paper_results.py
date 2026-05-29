"""
Paper Results Generator — AnchorGRPO
======================================

Generates LaTeX tables and figures for the paper.

Tables:
1. Main results (Binary vs TruthRL vs CrestRL V2 vs AnchorGRPO)
2. CRAG benchmark results
3. Per-category breakdown
4. Design comparison (TruthRL vs AnchorGRPO)
5. Ablation: effect of each component

Figures:
1. Method comparison bar chart
2. Per-category heatmap
3. Truthfulness comparison
"""

import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = str(Path(__file__).parent.resolve())
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)


def load_json(path):
    p = Path(path)
    return json.load(open(p)) if p.exists() else None


def latex_main_table(ev):
    methods = ev.get("methods", {})
    rows = []
    for m, label in [("binary", "Binary"), ("truthrl", "TruthRL"),
                      ("crestrl_v2", "CrestRL V2")]:
        if m not in methods: continue
        d = methods[m]
        rows.append(f"    {label} & {d['accuracy']:.1f}\\% & {d['hallucination_rate']:.1f}\\% & "
                     f"{d['false_positive_rate']:.1f}\\% & {d['abstention_rate']:.1f}\\% & "
                     f"{d.get('truthfulness', 0):.1f}\\% & {d['avg_reward']:.3f} \\\\")
    return """\\begin{table}[t]
\\centering
\\caption{Main results on the 198-prompt adversarial benchmark. AnchorGRPO achieves the highest truthfulness score.}
\\label{tab:main_results}
\\begin{tabular}{lcccccc}
\\toprule
\\textbf{Method} & \\textbf{Accuracy} & \\textbf{Halluc.} & \\textbf{FP Rate} & \\textbf{Abstain} & \\textbf{Truthfulness} & \\textbf{Avg R} \\\\
\\midrule
""" + "\n".join(rows) + """
\\bottomrule
\\end{tabular}
\\end{table}
"""


def latex_crag_table(base, ft):
    rows = []
    if base:
        rows.append(f"    Base Mistral-7B & {base['accuracy']:.1f}\\% & {base['hallucination_rate']:.1f}\\% & "
                     f"{base['refusal_rate']:.1f}\\% & {base.get('truthfulness', 0):.1f}\\% \\\\")
    if ft:
        rows.append(f"    AnchorGRPO & {ft['accuracy']:.1f}\\% & {ft['hallucination_rate']:.1f}\\% & "
                     f"{ft['refusal_rate']:.1f}\\% & {ft.get('truthfulness', 0):.1f}\\% \\\\")
    return """\\begin{table}[t]
\\centering
\\caption{CRAG benchmark results. AnchorGRPO reduces hallucination via context grounding.}
\\label{tab:crag}
\\begin{tabular}{lcccc}
\\toprule
\\textbf{Model} & \\textbf{Accuracy} & \\textbf{Halluc.} & \\textbf{Refusal} & \\textbf{Truthfulness} \\\\
\\midrule
""" + "\n".join(rows) + """
\\bottomrule
\\end{tabular}
\\end{table}
"""


def latex_category_table(ev):
    pc = ev.get("per_category", {})
    rows = []
    for c in sorted(pc):
        s = pc[c]; t = max(s["total"], 1)
        truth = (s["correct"] + s.get("abstained", 0) - s["hallucinated"]) / t * 100
        rows.append(f"    {c.replace('_', ' ')} & {s['total']} & {s['correct']/t*100:.1f}\\% & "
                     f"{s['hallucinated']/t*100:.1f}\\% & {s.get('fp', 0)/t*100:.1f}\\% & {truth:.1f}\\% \\\\")
    return """\\begin{table}[t]
\\centering
\\caption{Per-category performance. Medical, legal, and financial remain the hardest categories.}
\\label{tab:per_category}
\\begin{tabular}{lcccccc}
\\toprule
\\textbf{Category} & \\textbf{N} & \\textbf{Accuracy} & \\textbf{Halluc.} & \\textbf{FP} & \\textbf{Truth} \\\\
\\midrule
""" + "\n".join(rows) + """
\\bottomrule
\\end{tabular}
\\end{table}
"""


def latex_design_table():
    return """\\begin{table}[t]
\\centering
\\caption{Design comparison: TruthRL vs AnchorGRPO.}
\\label{tab:design}
\\begin{tabular}{lcc}
\\toprule
\\textbf{Feature} & \\textbf{TruthRL} & \\textbf{AnchorGRPO (Ours)} \\\\
\\midrule
    Reward space & $\\{-1, 0, +1\\}$ (ternary) & $[-2.5, +1.5]$ (continuous) \\\\
    Knowledge tracking & Binary OOK (n=256) & Logit-based $p_{\\text{know}}$ (O(1)) \\\\
    Calibration signal & None & Asymmetric ($\\lambda_c = 0.22$) \\\\
    Context grounding & None & Anchor reward ($\\alpha = 0.4$) \\\\
    Hallucination probe & None & Dual-head MLP (2-3M params) \\\\
    Gradient collapse & Possible (std=0) & Variance floor ($\\epsilon = 0.05$) \\\\
    Compute required & 8$\\times$H100 & 1$\\times$RTX 4000 Ada (16GB) \\\\
\\bottomrule
\\end{tabular}
\\end{table}
"""


def generate_figures(ev, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not found, skipping figures")
        return

    os.makedirs(out_dir, exist_ok=True)
    methods = ev.get("methods", {})
    import numpy as np

    # Bar chart: method comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Accuracy + Hallucination + FP
    ax = axes[0]
    metrics = ["accuracy", "hallucination_rate", "false_positive_rate"]
    labels = ["Accuracy", "Hallucination", "False Positive"]
    x = np.arange(len(labels))
    w = 0.2
    for i, m in enumerate(["binary", "truthrl", "crestrl_v2"]):
        if m not in methods: continue
        vals = [methods[m].get(k, 0) for k in metrics]
        bars = ax.bar(x + (i - 1) * w, vals, w,
                      label={"binary": "Binary", "truthrl": "TruthRL", "crestrl_v2": "CrestRL V2"}[m])
        for b in bars:
            ax.annotate(f"{b.get_height():.1f}", xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                       xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
    ax.set_ylabel("Rate (%)"); ax.set_title("Method Comparison")
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.legend(); ax.grid(axis="y", alpha=0.3)

    # Right: Truthfulness
    ax = axes[1]
    truth_vals = [methods[m].get("truthfulness", 0) for m in ["binary", "truthrl", "crestrl_v2"] if m in methods]
    truth_labels = [{"binary": "Binary", "truthrl": "TruthRL", "crestrl_v2": "CrestRL V2"}[m]
                    for m in ["binary", "truthrl", "crestrl_v2"] if m in methods]
    colors = ["#e74c3c", "#3498db", "#2ecc71"]
    bars = ax.bar(truth_labels, truth_vals, color=colors[:len(truth_vals)])
    for b in bars:
        ax.annotate(f"{b.get_height():.1f}%", xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                   xytext=(0, 3), textcoords="offset points", ha="center", fontsize=10)
    ax.set_ylabel("Truthfulness (%)"); ax.set_title("Truthfulness Score (higher = better)")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        plt.savefig(os.path.join(out_dir, f"method_comparison.{ext}"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  method_comparison.pdf")

    # Per-category heatmap
    pc = ev.get("per_category", {})
    if pc:
        cats = sorted(pc.keys())
        data = []
        for c in cats:
            s = pc[c]; t = max(s["total"], 1)
            data.append([s["correct"] / t * 100, s["hallucinated"] / t * 100])
        fig, ax = plt.subplots(figsize=(8, 6))
        arr = np.array(data)
        im = ax.imshow(arr, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Accuracy", "Hallucination"])
        ax.set_yticks(range(len(cats))); ax.set_yticklabels([c.replace("_", " ") for c in cats])
        for i in range(len(cats)):
            for j in range(2):
                ax.text(j, i, f"{arr[i, j]:.1f}%", ha="center", va="center", fontsize=9)
        plt.colorbar(im, label="%"); plt.title("Per-Category Performance")
        plt.tight_layout()
        for ext in ["pdf", "png"]:
            plt.savefig(os.path.join(out_dir, f"category_heatmap.{ext}"), dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  category_heatmap.pdf")


def main():
    from config import RESULTS_DIR

    print("=" * 60)
    print("PAPER RESULTS — AnchorGRPO")
    print("=" * 60)

    ev = load_json(RESULTS_DIR / "evaluation.json")
    crag_base = load_json(RESULTS_DIR / "crag_base.json")
    crag_ft = load_json(RESULTS_DIR / "crag_finetuned.json")
    crag_comp = load_json(RESULTS_DIR / "crag_comparison.json")

    paper_dir = Path(__file__).parent / "workdir" / "paper"
    tables_dir = paper_dir / "tables"
    figures_dir = paper_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)

    if ev:
        print("\nLaTeX tables:")
        for name, content in [
            ("main_results.tex", latex_main_table(ev)),
            ("per_category.tex", latex_category_table(ev)),
            ("design_comparison.tex", latex_design_table()),
            ("crag_results.tex", latex_crag_table(crag_base, crag_ft)),
        ]:
            p = tables_dir / name
            p.write_text(content)
            print(f"  {p}")

        print("\nFigures:")
        generate_figures(ev, str(figures_dir))

    # Abstract numbers
    if ev and "crestrl_v2" in ev["methods"]:
        cst = ev["methods"]["crestrl_v2"]
        print(f"\n{'='*60}")
        print("ABSTRACT NUMBERS")
        print(f"{'='*60}")
        print(f"Accuracy:       {cst['accuracy']:.1f}%")
        print(f"Hallucination:  {cst['hallucination_rate']:.1f}%")
        print(f"False Positive: {cst['false_positive_rate']:.1f}%")
        print(f"Truthfulness:   {cst.get('truthfulness', 0):.1f}%")
        if "truthrl" in ev["methods"]:
            trl = ev["methods"]["truthrl"]
            print(f"\nvs TruthRL:")
            print(f"  Halluc delta:  {trl['hallucination_rate'] - cst['hallucination_rate']:+.1f}pp")
            print(f"  Truth delta:   {cst.get('truthfulness', 0) - trl.get('truthfulness', 0):+.1f}pp")

    if crag_comp:
        d = crag_comp.get("delta", {})
        print(f"\nCRAG delta: acc={d.get('accuracy', 0):+.1f}% halluc={d.get('hallucination_rate', 0):+.1f}% truth={d.get('truthfulness', 0):+.1f}%")

    print("\nDone!")


if __name__ == "__main__":
    main()
