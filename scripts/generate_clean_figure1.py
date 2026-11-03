"""
Generate Clean, Visual, Minimal-Text Publication-Grade Figure 1 for Elsevier Manuscript.
Decluttered: Replaces dense text blocks with visual badges, clean cards, bold key metrics, and generous breathing room.
Canvas: 7.5 x 5.85 inches, calibrated for Elsevier single-page layout.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# Configure matplotlib for TrueType fonts and high quality
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Times']

def create_figure1(output_pdf_paths, output_png_paths):
    fig = plt.figure(figsize=(7.5, 5.85), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 7.5)
    ax.set_ylim(0, 5.85)
    ax.axis('off')

    # Color Palette - Professional Editorial
    c_navy = "#1e3a8a"      # Headers / Primary
    c_blue = "#2563eb"      # Focus blue
    c_blue_bg = "#f0f7ff"   # Soft blue fill
    c_blue_border = "#93c5fd"
    c_slate = "#0f172a"     # Main dark text
    c_muted = "#475569"     # Secondary text
    c_gray_bg = "#f8fafc"   # Neutral fill
    c_gray_border = "#cbd5e1"
    c_emerald = "#059669"   # Positive / Performance
    c_emerald_bg = "#ecfdf5"
    c_emerald_border = "#a7f3d0"
    c_amber = "#b45309"     # Validation
    c_amber_bg = "#fffbeb"
    c_amber_border = "#fde68a"
    c_red = "#dc2626"       # Test / Strict
    c_red_bg = "#fef2f2"
    c_red_border = "#fecaca"

    def draw_card(x, y, w, h, bg, border, lw=1.0, rad=0.06):
        box = FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad=0.0,rounding_size={rad}",
            facecolor=bg, edgecolor=border,
            linewidth=lw, zorder=1
        )
        ax.add_patch(box)
        return box

    def draw_arrow(x1, y1, x2, y2, color=c_blue, lw=1.3):
        ax.annotate(
            '', xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle="-|>", color=color, lw=lw,
                mutation_scale=10, shrinkA=0, shrinkB=0
            ),
            zorder=6
        )

    # =========================================================================
    # ROW 1: PANEL (a) END-TO-END WORKFLOW (y: 3.92 to 5.75)
    # =========================================================================
    draw_card(0.12, 3.92, 7.26, 1.83, "#ffffff", c_gray_border, lw=1.1, rad=0.08)
    
    ax.text(0.25, 5.62, "(a) End-to-End Prediction Workflow",
            fontsize=10.5, fontweight='bold', color=c_navy, va='center')
    ax.text(7.25, 5.62, "Monthly Forward-Time Forecasting",
            fontsize=8.0, fontstyle='italic', color=c_muted, ha='right', va='center')

    # Stage 1: Heterogeneous Inputs (x: 0.22, w: 1.55, y: 4.02, h: 1.44)
    draw_card(0.22, 4.02, 1.55, 1.44, c_gray_bg, c_gray_border, lw=0.8)
    ax.text(1.00, 5.32, "Input Data Sources", fontsize=8.6, fontweight='bold', color=c_navy, ha='center')

    # 3 visual source pills
    draw_card(0.30, 4.88, 1.39, 0.30, "#ffffff", "#e2e8f0", lw=0.7)
    ax.text(1.00, 5.03, "Environment & Climate (18)", fontsize=7.6, fontweight='bold', color=c_slate, ha='center', va='center')

    draw_card(0.30, 4.50, 1.39, 0.30, "#ffffff", "#e2e8f0", lw=0.7)
    ax.text(1.00, 4.65, "Host & Infrastructure (8)", fontsize=7.6, fontweight='bold', color=c_slate, ha='center', va='center')

    draw_card(0.30, 4.12, 1.39, 0.30, "#ffffff", "#e2e8f0", lw=0.7)
    ax.text(1.00, 4.27, "Surveillance & Lags (36)", fontsize=7.6, fontweight='bold', color=c_slate, ha='center', va='center')

    draw_arrow(1.77, 4.74, 2.00, 4.74)

    # Stage 2: Feature Vector (x: 2.00, w: 1.25, y: 4.02, h: 1.44)
    draw_card(2.00, 4.02, 1.25, 1.44, c_blue_bg, c_blue_border, lw=0.8)
    ax.text(2.62, 5.32, "CSD Feature Vector", fontsize=8.6, fontweight='bold', color=c_navy, ha='center')
    ax.text(2.62, 5.08, r"$\mathbf{x}_{i,t} \in \mathbb{R}^{62}$", fontsize=11.0, fontweight='bold', color=c_blue, ha='center')
    
    draw_card(2.10, 4.54, 1.05, 0.34, "#ffffff", c_blue_border, lw=0.6)
    ax.text(2.62, 4.71, "43 Non-Lag Features", fontsize=7.2, fontweight='bold', color=c_slate, ha='center', va='center')

    draw_card(2.10, 4.12, 1.05, 0.34, "#ffffff", c_blue_border, lw=0.6)
    ax.text(2.62, 4.29, "19 Outbreak Lags", fontsize=7.2, fontweight='bold', color=c_slate, ha='center', va='center')

    draw_arrow(3.25, 4.74, 3.48, 4.74)

    # Stage 3: Spatial GraphSAGE (x: 3.48, w: 1.76, y: 4.02, h: 1.44)
    draw_card(3.48, 4.02, 1.76, 1.44, "#ffffff", c_blue_border, lw=0.8)
    ax.text(4.36, 5.32, "Spatial GraphSAGE", fontsize=8.6, fontweight='bold', color=c_navy, ha='center')
    ax.text(4.36, 5.16, r"$k\mathrm{NN}-3$ Centroid Graph (5,054 CSDs)", fontsize=7.2, color=c_muted, ha='center')

    draw_card(3.58, 4.62, 1.56, 0.42, c_blue_bg, c_blue_border, lw=0.6)
    ax.text(4.36, 4.90, r"Layer 1: $62 \to 64$ (ReLU)", fontsize=7.8, fontweight='bold', color=c_blue, ha='center')
    ax.text(4.36, 4.74, r"$\mathbf{h}_i^{(1)} = \sigma(W_1 [\mathbf{x}_i \,\|\, \bar{\mathbf{x}}_{\mathcal{N}(i)}])$", fontsize=7.4, color=c_slate, ha='center')

    draw_card(3.58, 4.12, 1.56, 0.42, c_blue_bg, c_blue_border, lw=0.6)
    ax.text(4.36, 4.40, r"Layer 2: $64 \to 64$ (ReLU)", fontsize=7.8, fontweight='bold', color=c_blue, ha='center')
    ax.text(4.36, 4.24, r"$\mathbf{z}_i = \sigma(W_2 [\mathbf{h}_i^{(1)} \,\|\, \bar{\mathbf{h}}_{\mathcal{N}(i)}^{(1)}])$", fontsize=7.4, color=c_slate, ha='center')

    draw_arrow(5.24, 4.74, 5.46, 4.74)

    # Stage 4: Head (x: 5.46, w: 0.94, y: 4.02, h: 1.44)
    draw_card(5.46, 4.02, 0.94, 1.44, c_gray_bg, c_gray_border, lw=0.8)
    ax.text(5.93, 5.32, "Head", fontsize=8.6, fontweight='bold', color=c_navy, ha='center')

    draw_card(5.53, 4.68, 0.80, 0.46, "#ffffff", c_gray_border, lw=0.6)
    ax.text(5.93, 4.99, "Embedding", fontsize=6.8, color=c_muted, ha='center')
    ax.text(5.93, 4.81, r"$\mathbf{z}_{i,t} \in \mathbb{R}^{64}$", fontsize=8.0, fontweight='bold', color=c_blue, ha='center')

    draw_card(5.53, 4.12, 0.80, 0.46, "#ffffff", c_gray_border, lw=0.6)
    ax.text(5.93, 4.44, r"Linear $64 \to 1$", fontsize=7.2, fontweight='bold', color=c_slate, ha='center')
    ax.text(5.93, 4.25, r"Sigmoid $\sigma$", fontsize=7.2, color=c_muted, ha='center')

    draw_arrow(6.40, 4.74, 6.62, 4.74)

    # Stage 5: Target Risk (x: 6.62, w: 0.66, y: 4.02, h: 1.44)
    draw_card(6.62, 4.02, 0.66, 1.44, c_emerald_bg, c_emerald_border, lw=1.1)
    ax.text(6.95, 5.32, "Risk Alert", fontsize=8.6, fontweight='bold', color=c_emerald, ha='center')
    ax.text(6.95, 4.94, r"$\hat{p}_{i,t+1}$", fontsize=12.0, fontweight='bold', color=c_emerald, ha='center')
    ax.text(6.95, 4.60, r"$P(Y=1)$", fontsize=7.2, color=c_slate, ha='center')
    ax.text(6.95, 4.34, "Month", fontsize=6.8, color=c_muted, ha='center')
    ax.text(6.95, 4.19, "$t+1$", fontsize=8.6, fontweight='bold', color=c_slate, ha='center')


    # =========================================================================
    # ROW 2: PANELS (b) & (c) (y: 2.01 to 3.84)
    # =========================================================================
    # Panel (b): Feature Space Breakdown (x: 0.12 to 3.68)
    draw_card(0.12, 2.01, 3.56, 1.83, "#ffffff", c_gray_border, lw=1.1, rad=0.08)
    ax.text(0.25, 3.71, "(b) Feature Space Composition (62 Features)",
            fontsize=9.8, fontweight='bold', color=c_navy, va='center')

    # Non-lag card
    draw_card(0.22, 2.76, 3.36, 0.78, c_blue_bg, c_blue_border, lw=0.7)
    ax.text(0.32, 3.41, "43 Non-Lag Predictors (Landscape & Agricultural Context)", fontsize=7.8, fontweight='bold', color=c_navy)
    
    pills_b1 = [
        ("Land Cover: 10", "Dynamic World"),
        ("Climate: 8", "ERA5-Land"),
        ("Wild Birds: 17", "eBird & PCR"),
        ("Hosts & Roads: 8", "Poultry & Hwy")
    ]
    for idx, (p_title, p_sub) in enumerate(pills_b1):
        px = 0.32 + (idx % 2) * 1.68
        py = 3.12 if idx < 2 else 2.84
        draw_card(px, py, 1.58, 0.25, "#ffffff", c_blue_border, lw=0.5, rad=0.04)
        ax.text(px + 0.08, py + 0.12, p_title, fontsize=7.0, fontweight='bold', color=c_slate, va='center')
        ax.text(px + 1.50, py + 0.12, p_sub, fontsize=6.4, color=c_muted, ha='right', va='center')

    # Lags card
    draw_card(0.22, 2.08, 3.36, 0.62, c_red_bg, c_red_border, lw=0.7)
    ax.text(0.32, 2.58, "19 Outbreak Lags (Surveillance Memory)", fontsize=7.8, fontweight='bold', color=c_red)
    
    pills_b2 = [
        ("CSD Lags: 4", "t-1, t-2, t-3, t-12"),
        ("Provincial Lags: 12", "1 to 12 months"),
        ("Spatial Pressure: 3", "Neighbor cases"),
        ("Strict Cutoff", "Zero leakage <= t")
    ]
    for idx, (p_title, p_sub) in enumerate(pills_b2):
        px = 0.32 + (idx % 2) * 1.68
        py = 2.33 if idx < 2 else 2.13
        draw_card(px, py, 1.58, 0.18, "#ffffff", c_red_border, lw=0.5, rad=0.04)
        ax.text(px + 0.08, py + 0.09, p_title, fontsize=6.8, fontweight='bold', color=c_slate, va='center')
        ax.text(px + 1.50, py + 0.09, p_sub, fontsize=6.2, color=c_muted, ha='right', va='center')


    # Panel (c): Graph Message Passing (x: 3.82 to 7.38)
    draw_card(3.82, 2.01, 3.56, 1.83, "#ffffff", c_gray_border, lw=1.1, rad=0.08)
    ax.text(3.95, 3.71, r"(c) Spatial GraphSAGE Message Passing ($k\mathrm{NN}-3$)",
            fontsize=9.8, fontweight='bold', color=c_navy, va='center')

    # Left: Centroid Topology Diagram
    draw_card(3.92, 2.08, 1.34, 1.46, c_gray_bg, c_gray_border, lw=0.7)
    ax.text(4.59, 3.41, "Centroid Graph", fontsize=7.6, fontweight='bold', color=c_navy, ha='center')

    node_i = (4.59, 2.72)
    node_j1 = (4.18, 3.10)
    node_j2 = (5.00, 3.10)
    node_j3 = (4.59, 2.25)

    for nj, label in [(node_j1, r"$j_1$"), (node_j2, r"$j_2$"), (node_j3, r"$j_3$")]:
        ax.plot([node_i[0], nj[0]], [node_i[1], nj[1]], linestyle='--', lw=1.0, color="#94a3b8", zorder=3)
        circ = plt.Circle(nj, 0.10, facecolor="#ffffff", edgecolor=c_blue, lw=1.1, zorder=4)
        ax.add_patch(circ)
        ax.text(nj[0], nj[1], label, fontsize=7.3, fontweight='bold', color=c_blue, ha='center', va='center', zorder=5)

    circ_i = plt.Circle(node_i, 0.13, facecolor=c_blue, edgecolor=c_navy, lw=1.4, zorder=4)
    ax.add_patch(circ_i)
    ax.text(node_i[0], node_i[1], r"$i$", fontsize=8.8, fontweight='bold', color="#ffffff", ha='center', va='center', zorder=5)

    ax.text(4.59, 2.45, "Mean Pooling", fontsize=6.6, fontstyle='italic', color=c_muted, ha='center')

    # Right: Mathematical Formulations
    draw_card(5.32, 2.82, 1.96, 0.72, c_blue_bg, c_blue_border, lw=0.7)
    ax.text(6.30, 3.41, "Neighborhood Aggregation", fontsize=7.6, fontweight='bold', color=c_navy, ha='center')
    ax.text(6.30, 3.20, r"$\bar{\mathbf{h}}_{\mathcal{N}(i)} = \frac{1}{3} \sum_{j \in \mathcal{N}(i)} \mathbf{h}_j$",
            fontsize=8.0, color=c_slate, ha='center')
    ax.text(6.30, 2.95, r"$\mathbf{h}_i^{(l)} = \sigma(W_l [\mathbf{h}_i^{(l-1)} \,\|\, \bar{\mathbf{h}}_{\mathcal{N}(i)}])$",
            fontsize=8.0, fontweight='bold', color=c_blue, ha='center')

    draw_card(5.32, 2.08, 1.96, 0.68, "#ffffff", c_gray_border, lw=0.7)
    ax.text(5.42, 2.62, "Inductive Spatial Benefits:", fontsize=7.4, fontweight='bold', color=c_navy, ha='left')
    ax.text(5.42, 2.45, "• Pools regional exposure across borders", fontsize=6.8, color=c_slate, ha='left')
    ax.text(5.42, 2.31, "• Sparse kNN-3 prevents oversmoothing", fontsize=6.8, color=c_slate, ha='left')
    ax.text(5.42, 2.17, "• 34.2% attribution from spatial context", fontsize=6.8, color=c_slate, ha='left')


    # =========================================================================
    # ROW 3: PANELS (d) & (e) (y: 0.10 to 1.93)
    # =========================================================================
    # Panel (d): Forward-Time Splitting (x: 0.12 to 3.68)
    draw_card(0.12, 0.10, 3.56, 1.83, "#ffffff", c_gray_border, lw=1.1, rad=0.08)
    ax.text(0.25, 1.80, "(d) Prospective Forward-Time Splitting",
            fontsize=9.8, fontweight='bold', color=c_navy, va='center')

    # Rolling rule banner
    draw_card(0.22, 1.36, 3.36, 0.26, c_gray_bg, c_gray_border, lw=0.6)
    ax.text(1.90, 1.49, r"Rolling Setup: Features $\leq t$  $\longrightarrow$  Forecast $t+1$ (No Leakage)",
            fontsize=7.3, fontweight='bold', color=c_navy, ha='center', va='center')

    # 3 Partition cards
    # Train
    draw_card(0.22, 0.94, 3.36, 0.36, c_blue_bg, c_blue_border, lw=0.7)
    ax.text(0.32, 1.18, "Training Set (2022–2023)", fontsize=7.6, fontweight='bold', color=c_blue)
    ax.text(3.48, 1.18, "24 Mo  |  121,296 CSD-mo", fontsize=7.0, color=c_muted, ha='right')
    ax.text(0.32, 1.03, "262 Outbreak Events (0.2160% prev.)  •  Fit scalers exclusively here", fontsize=6.6, color=c_slate)

    # Val
    draw_card(0.22, 0.54, 3.36, 0.36, c_amber_bg, c_amber_border, lw=0.7)
    ax.text(0.32, 0.78, "Validation Set (2024)", fontsize=7.6, fontweight='bold', color=c_amber)
    ax.text(3.48, 0.78, "12 Mo  |  60,648 CSD-mo", fontsize=7.0, color=c_muted, ha='right')
    ax.text(0.32, 0.63, "40 Outbreak Events (0.0660% prev.)  •  Model selection & early stopping", fontsize=6.6, color=c_slate)

    # Test
    draw_card(0.22, 0.14, 3.36, 0.36, c_red_bg, c_red_border, lw=0.7)
    ax.text(0.32, 0.38, "Locked Test Set (2025)", fontsize=7.6, fontweight='bold', color=c_red)
    ax.text(3.48, 0.38, "11 Mo  |  55,594 CSD-mo", fontsize=7.0, color=c_muted, ha='right')
    ax.text(0.32, 0.23, "79 Outbreak Events (0.1421% prev.)  •  Prospective frozen evaluation", fontsize=6.6, color=c_slate)


    # Panel (e): Performance Lift (x: 3.82 to 7.38)
    draw_card(3.82, 0.10, 3.56, 1.83, "#ffffff", c_gray_border, lw=1.1, rad=0.08)
    ax.text(3.95, 1.80, "(e) Spatial Performance Lift (GraphSAGE vs. MLP)",
            fontsize=9.8, fontweight='bold', color=c_navy, va='center')

    # Card 1: Full Model
    draw_card(3.92, 0.94, 3.36, 0.68, c_emerald_bg, c_emerald_border, lw=0.8)
    ax.text(4.02, 1.50, "Full Model (62 Features): GraphSAGE vs. Matched MLP", fontsize=7.6, fontweight='bold', color=c_emerald)
    
    rows_e1 = [
        ("1:5 Benchmark:", "0.841 vs 0.808", "+0.033 (98%)"),
        ("Commercial:", "0.107 vs 0.081", "+0.025 PR-AUC"),
        ("National:", "0.111 vs 0.070", "+0.040 (95% CI > 0)")
    ]
    for idx, (regime, vals, lift) in enumerate(rows_e1):
        ry = 1.34 - idx * 0.16
        ax.text(4.02, ry, regime, fontsize=6.8, fontweight='bold', color=c_slate)
        ax.text(5.30, ry, vals, fontsize=6.8, color=c_slate)
        ax.text(7.18, ry, lift, fontsize=6.8, fontweight='bold', color=c_emerald, ha='right')

    # Card 2: No-Lag Ablation
    draw_card(3.92, 0.14, 3.36, 0.74, c_red_bg, c_red_border, lw=0.8)
    ax.text(4.02, 0.76, "Strict No-Lag Ablation (43 Features, Zero History Lags)", fontsize=7.6, fontweight='bold', color=c_red)

    rows_e2 = [
        ("1:5 Benchmark:", "0.782 vs 0.728", "+0.054 (100%)"),
        ("Commercial:", "0.091 vs 0.069", "+0.023 PR-AUC"),
        ("National:", "0.076 vs 0.046", "+0.030 (95% CI > 0)")
    ]
    for idx, (regime, vals, lift) in enumerate(rows_e2):
        ry = 0.60 - idx * 0.16
        ax.text(4.02, ry, regime, fontsize=6.8, fontweight='bold', color=c_slate)
        ax.text(5.30, ry, vals, fontsize=6.8, color=c_slate)
        ax.text(7.18, ry, lift, fontsize=6.8, fontweight='bold', color=c_red, ha='right')

    # Save to all target paths
    for p in output_pdf_paths:
        fig.savefig(p, format='pdf', dpi=300, bbox_inches='tight')
        print(f"Saved PDF: {p}")

    for p in output_png_paths:
        fig.savefig(p, format='png', dpi=300, bbox_inches='tight')
        print(f"Saved PNG: {p}")

    plt.close(fig)

if __name__ == "__main__":
    pdf_paths = [
        r"d:\gdrive\models\hpai-heterogeneous-gnn-canada\results\manuscript_elsevier\figures\Figure1_schematic.pdf",
        r"d:\gdrive\models\hpai-heterogeneous-gnn-canada\results\manuscript_elsevier\Figure1_schematic.pdf",
        r"d:\gdrive\models\hpai-heterogeneous-gnn-canada\results\manuscript_elsevier\Figure1_study_design.pdf",
        r"d:\gdrive\models\hpai-heterogeneous-gnn-canada\results\manuscript_latex\figures\Figure1_schematic.pdf",
        r"d:\gdrive\models\hpai-heterogeneous-gnn-canada\results\manuscript_latex\Figure1_schematic.pdf"
    ]
    png_paths = [
        r"d:\gdrive\models\hpai-heterogeneous-gnn-canada\results\manuscript_elsevier\figures\Figure1_schematic.png",
        r"d:\gdrive\models\hpai-heterogeneous-gnn-canada\results\manuscript_elsevier\Figure1_schematic.png",
        r"d:\gdrive\models\hpai-heterogeneous-gnn-canada\results\manuscript_elsevier\Figure1_study_design.png",
        r"d:\gdrive\models\hpai-heterogeneous-gnn-canada\results\manuscript_latex\figures\Figure1_schematic.png",
        r"d:\gdrive\models\hpai-heterogeneous-gnn-canada\results\manuscript_latex\Figure1_schematic.png"
    ]
    create_figure1(pdf_paths, png_paths)
