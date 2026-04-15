"""
Experiment Evaluation Summary Generator

Automatically scans .workspace/experiments/*/eval/summary_metrics.json,
generates a Markdown summary report for review by other agents or humans.

Usage:
  python summarize.py
  python summarize.py --experiments-dir .workspace/experiments
  python summarize.py --output .workspace/experiments/SUMMARY.md
"""

import json
import os
import sys
import argparse
import datetime
from pathlib import Path


# Parameter name -> LaTeX display name
PARAM_DISPLAY = {
    "r":          r"log₁₀ r",
    "n_t":        r"n_t",
    "kappa10":    r"log₁₀ κ₁₀",
    "T_re":       r"log₁₀ T_re",
    "DN_re":      r"ΔN_re",
    "Omega_bh2":  r"Ω_b h²",
    "Omega_ch2":  r"Ω_c h²",
    "H0":         r"H₀",
    "A_s":        r"log(10¹⁰ A_s)",
}

# Task ordering
TASK_ORDER = ["PTA", "LISA", "LIGO", "joint"]
TASK_DISPLAY = {"PTA": "PTA", "LISA": "LISA", "LIGO": "LIGO", "joint": "Joint"}


def discover_experiments(experiments_dir):
    """Scan all subdirectories under experiments_dir that contain eval/summary_metrics.json."""
    results = []
    exp_dir = Path(experiments_dir)
    if not exp_dir.is_dir():
        return results

    for sub in sorted(exp_dir.iterdir()):
        summary_path = sub / "eval" / "summary_metrics.json"
        if summary_path.is_file():
            results.append((sub.name, str(summary_path)))
    return results


def load_summary(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def fmt(val, digits=4):
    """Format a floating-point number."""
    if val is None:
        return "N/A"
    return f"{val:.{digits}f}"


def render_per_experiment(name, data):
    """Render a detailed report for a single experiment."""
    lines = []
    lines.append(f"### {name}\n")
    lines.append(f"- **Timestamp**: {data.get('timestamp', 'N/A')}")
    lines.append(f"- **Num posterior samples**: {data.get('num_samples', 'N/A')}")
    lines.append(f"- **Tasks evaluated**: {', '.join(data.get('tasks_evaluated', []))}")
    lines.append("")

    per_task = data.get("per_task_metrics", {})
    tasks = [t for t in TASK_ORDER if t in per_task]

    # ---- Average metrics per task ----
    lines.append("#### Average Metrics by Task\n")
    lines.append(f"| Task | R² | NMAE (%) | Rel. CI Width | Samples |")
    lines.append(f"|------|----|----------|---------------|---------|")
    for task in tasks:
        m = per_task[task]
        lines.append(
            f"| {TASK_DISPLAY.get(task, task)} "
            f"| {fmt(m.get('avg_r2'))} "
            f"| {fmt(m.get('avg_nmae_pct'), 2)} "
            f"| {fmt(m.get('avg_rel_ci_width'))} "
            f"| {m.get('n_samples', 'N/A')} |"
        )
    lines.append("")

    # ---- R² matrix per parameter x task ----
    lines.append("#### R² by Parameter & Task\n")

    # Collect all parameter names (preserve order)
    param_names = []
    for task in tasks:
        for p in per_task[task].get("per_param", {}):
            if p not in param_names:
                param_names.append(p)

    header = "| Parameter | " + " | ".join(TASK_DISPLAY.get(t, t) for t in tasks) + " |"
    sep =    "|-----------|" + "|".join(["--------"] * len(tasks)) + "|"
    lines.append(header)
    lines.append(sep)

    for p in param_names:
        display = PARAM_DISPLAY.get(p, p)
        row = f"| {display} "
        for task in tasks:
            val = per_task[task].get("per_param", {}).get(p, {}).get("r2", None)
            row += f"| {fmt(val)} "
        row += "|"
        lines.append(row)
    lines.append("")

    # ---- NMAE matrix per parameter x task ----
    lines.append("#### NMAE (%) by Parameter & Task\n")
    header = "| Parameter | " + " | ".join(TASK_DISPLAY.get(t, t) for t in tasks) + " |"
    sep =    "|-----------|" + "|".join(["--------"] * len(tasks)) + "|"
    lines.append(header)
    lines.append(sep)

    for p in param_names:
        display = PARAM_DISPLAY.get(p, p)
        row = f"| {display} "
        for task in tasks:
            val = per_task[task].get("per_param", {}).get(p, {}).get("nmae_pct", None)
            row += f"| {fmt(val, 2)} "
        row += "|"
        lines.append(row)
    lines.append("")

    # ---- MAE per parameter ----
    lines.append("#### MAE by Parameter & Task\n")
    header = "| Parameter | " + " | ".join(TASK_DISPLAY.get(t, t) for t in tasks) + " |"
    sep =    "|-----------|" + "|".join(["--------"] * len(tasks)) + "|"
    lines.append(header)
    lines.append(sep)

    for p in param_names:
        display = PARAM_DISPLAY.get(p, p)
        row = f"| {display} "
        for task in tasks:
            val = per_task[task].get("per_param", {}).get(p, {}).get("mae", None)
            row += f"| {fmt(val)} "
        row += "|"
        lines.append(row)
    lines.append("")

    # ---- Rel. CI Width per parameter ----
    lines.append("#### Relative CI Width by Parameter & Task\n")
    header = "| Parameter | " + " | ".join(TASK_DISPLAY.get(t, t) for t in tasks) + " |"
    sep =    "|-----------|" + "|".join(["--------"] * len(tasks)) + "|"
    lines.append(header)
    lines.append(sep)

    for p in param_names:
        display = PARAM_DISPLAY.get(p, p)
        row = f"| {display} "
        for task in tasks:
            val = per_task[task].get("per_param", {}).get(p, {}).get("rel_ci_width", None)
            row += f"| {fmt(val)} "
        row += "|"
        lines.append(row)
    lines.append("")

    # ---- KS statistics ----
    lines.append("#### KS Statistic by Parameter & Task\n")
    header = "| Parameter | " + " | ".join(TASK_DISPLAY.get(t, t) for t in tasks) + " |"
    sep =    "|-----------|" + "|".join(["--------"] * len(tasks)) + "|"
    lines.append(header)
    lines.append(sep)

    for p in param_names:
        display = PARAM_DISPLAY.get(p, p)
        row = f"| {display} "
        for task in tasks:
            val = per_task[task].get("per_param", {}).get(p, {}).get("ks_stat", None)
            row += f"| {fmt(val)} "
        row += "|"
        lines.append(row)
    lines.append("")

    return "\n".join(lines)


def _cross_param_table(lines, title, exp_names, param_names, all_data, task, key, digits=4):
    """Render a cross-experiment per-parameter metric table."""
    lines.append(f"### {title}\n")
    header = "| Parameter | " + " | ".join(exp_names) + " |"
    sep =    "|-----------|" + "|".join(["--------"] * len(exp_names)) + "|"
    lines.append(header)
    lines.append(sep)
    for p in param_names:
        display = PARAM_DISPLAY.get(p, p)
        row = f"| {display} "
        for _, data in all_data:
            task_m = data.get("per_task_metrics", {}).get(task, {})
            val = task_m.get("per_param", {}).get(p, {}).get(key, None)
            row += f"| {fmt(val, digits)} "
        row += "|"
        lines.append(row)
    lines.append("")


def render_cross_comparison(all_data):
    """Cross-experiment comparison: output R² / MAE / Rel. CI Width per parameter for each task."""
    if len(all_data) < 2:
        return ""

    lines = []
    lines.append("## Cross-Experiment Comparison\n")

    # Collect parameters
    param_names = []
    for name, data in all_data:
        joint = data.get("per_task_metrics", {}).get("joint", {})
        for p in joint.get("per_param", {}):
            if p not in param_names:
                param_names.append(p)

    if not param_names:
        return ""

    exp_names = [n for n, _ in all_data]

    # For each task, output three tables: R², MAE, Rel. CI Width
    for task in ["joint", "PTA", "LISA", "LIGO"]:
        has_data = any(
            task in data.get("per_task_metrics", {})
            for _, data in all_data
        )
        if not has_data:
            continue

        task_label = TASK_DISPLAY.get(task, task)
        lines.append(f"## {task_label} — Cross-Experiment\n")

        _cross_param_table(lines, f"{task_label} R²", exp_names, param_names, all_data, task, "r2")
        _cross_param_table(lines, f"{task_label} MAE", exp_names, param_names, all_data, task, "mae")
        _cross_param_table(lines, f"{task_label} Relative CI Width", exp_names, param_names, all_data, task, "rel_ci_width")

    return "\n".join(lines)


def generate_summary(experiments_dir, output_path):
    """Main function: scan + render + write output."""
    discovered = discover_experiments(experiments_dir)

    if not discovered:
        print(f"No experiments found under {experiments_dir}")
        return

    print(f"Discovered {len(discovered)} experiment(s):")
    for name, path in discovered:
        print(f"  - {name}")

    # Load
    all_data = []
    for name, path in discovered:
        data = load_summary(path)
        all_data.append((name, data))

    # Render
    lines = []
    lines.append(f"# Experiment Evaluation Summary\n")
    lines.append(f"- **Generated**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- **Source**: `{experiments_dir}`")
    lines.append(f"- **Experiments**: {len(all_data)}")
    lines.append("")

    # Per-experiment detailed report
    for name, data in all_data:
        lines.append("---\n")
        lines.append(render_per_experiment(name, data))

    # Cross-experiment comparison
    cross = render_cross_comparison(all_data)
    if cross:
        lines.append("---\n")
        lines.append(cross)

    md_content = "\n".join(lines)

    # Write output
    out_dir = os.path.dirname(output_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"\nSummary written to {output_path}")
    print(f"Total lines: {len(md_content.splitlines())}")


def main():
    # Default to script directory as base
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))

    default_exp_dir = os.path.join(project_root, ".workspace", "experiments")
    default_output = os.path.join(project_root, ".workspace", "experiments", "SUMMARY.md")

    parser = argparse.ArgumentParser(description="Generate evaluation summary from experiments")
    parser.add_argument("--experiments-dir", type=str, default=default_exp_dir,
                        help="Root directory containing experiment packages "
                             "(default: .workspace/experiments/)")
    parser.add_argument("--output", type=str, default=default_output,
                        help="Output Markdown file path "
                             "(default: .workspace/experiments/SUMMARY.md)")
    args = parser.parse_args()

    generate_summary(args.experiments_dir, args.output)


if __name__ == "__main__":
    main()
