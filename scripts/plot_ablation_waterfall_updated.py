import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

csv_path = r'C:\Users\mkhos\PycharmProjects\Spatiotemporal_HPAI_Risk_Modeling\benchmark_results\modality_analysis\feature_modality_ablation_summary.csv'
df = pd.read_csv(csv_path)

# Select key scenarios to present cleanly
scenarios_to_include = [
    "Full Model (All Features)",
    "Satellite EO Only",
    "Full Model (No Contagion)",
    "Without Static Infrastructure",
    "Without Satellite EO",
    "Static Infrastructure Only"
]

df_plot = df[df["Scenario"].isin(scenarios_to_include)].copy()

# Rename scenarios for display
scenario_names = {
    "Full Model (All Features)": "Full Multi-Modal (97 vars)",
    "Satellite EO Only": "Satellite EO Only (53 vars)",
    "Full Model (No Contagion)": "Without Contagion Proximity (94 vars)",
    "Without Static Infrastructure": "Without Static Infra (92 vars)",
    "Without Satellite EO": "Without Satellite EO (44 vars)",
    "Static Infrastructure Only": "Static Infra Only (5 vars)"
}

df_plot["DisplayName"] = df_plot["Scenario"].map(scenario_names)
df_plot = df_plot.sort_values(by="Spatial_PR_AUC", ascending=False).reset_index(drop=True)

# Colors
colors = ["#1b9e77", "#2b5c8f", "#7570b3", "#386cb0", "#d95f02", "#e7298a"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5), dpi=300)

# Panel A: Spatial PR-AUC
bars1 = ax1.barh(
    df_plot["DisplayName"],
    df_plot["Spatial_PR_AUC"],
    xerr=df_plot["Spatial_PR_AUC_Std"],
    color=colors[:len(df_plot)],
    edgecolor="#212529",
    height=0.55,
    alpha=0.88,
    capsize=4
)
ax1.set_xlabel("Spatial PR-AUC (Precision-Recall AUC)", fontsize=11, fontweight="bold")
ax1.set_title("A. Impact of Modality Ablation on Spatial PR-AUC", fontsize=12, fontweight="bold", pad=12)
ax1.grid(True, axis="x", linestyle=":", alpha=0.5)
ax1.set_xlim([0.50, 1.02])
ax1.invert_yaxis()

for bar in bars1:
    w = bar.get_width()
    ax1.text(w + 0.015, bar.get_y() + bar.get_height() / 2, f"{w:.4f}", va="center", ha="left", fontsize=10, fontweight="bold")

# Panel B: Top-10% Sample Recall
bars2 = ax2.barh(
    df_plot["DisplayName"],
    df_plot["Outbreak_Capture_at_10pct"],
    color=colors[:len(df_plot)],
    edgecolor="#212529",
    height=0.55,
    alpha=0.88
)
ax2.set_xlabel("Top-10% Sample Outbreak Recall (%)", fontsize=11, fontweight="bold")
ax2.set_title("B. Operational Surveillance Efficiency: Top-10% Sample Recall", fontsize=12, fontweight="bold", pad=12)
ax2.grid(True, axis="x", linestyle=":", alpha=0.5)
ax2.set_xlim([0, 50])
ax2.invert_yaxis()

for bar in bars2:
    w = bar.get_width()
    ax2.text(w + 1.0, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%", va="center", ha="left", fontsize=10, fontweight="bold")

plt.tight_layout()

out_paths = [
    r'C:\Users\mkhos\PycharmProjects\Spatiotemporal_HPAI_Risk_Modeling\benchmark_results\modality_analysis\modality_ablation_performance_waterfall.png',
    r'C:\Users\mkhos\PycharmProjects\Spatiotemporal_HPAI_Risk_Modeling\manuscript_elsevier_eswa\figures\fig5_modality_ablation_waterfall.png'
]

for p in out_paths:
    os.makedirs(os.path.dirname(p), exist_ok=True)
    plt.savefig(p, dpi=300, bbox_inches="tight")
    print(f"Saved updated figure to: {p}")

plt.close()
