"""
HPAI Spatial GraphSAGE Architecture — Publication-Quality Scientific Schematic
Generates a multi-panel journal-style architecture figure for:
"Quantifying the Predictive Value of Spatial Graph Learning for One-Month-Ahead HPAI Poultry Outbreak Risk Across Canada"

Visual Style:
- Clean modular scientific blocks with rounded rectangles (FancyBboxPatch)
- Restrained academic pastel palette
- Thin 0.8-1.2 pt gray/black outlines
- No 3D effects, shadows, or gradients
- Clear multi-panel layout (a, b, c, d, e)
- High-resolution export to SVG, PDF, and PNG (300 DPI)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch, ArrowStyle
import numpy as np
from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR_LATEX = ROOT / "results/manuscript_latex/figures"
OUT_DIR_FINAL = ROOT / "results/manuscript_draft/figures/final"
OUT_DIR_LATEX.mkdir(parents=True, exist_ok=True)
OUT_DIR_FINAL.mkdir(parents=True, exist_ok=True)

# Color Palette (Pastel Academic)
C_ENV = "#E3F2FD"      # Pale Blue (Remote sensing / Dynamic World)
C_CLIM = "#E0F7FA"     # Pale Cyan (ERA5-Land Climate)
C_HOST = "#FFF3E0"     # Pale Orange (GLW Host Densities)
C_BIRD = "#F3E5F5"     # Pale Purple (Wild-bird detections)
C_ROAD = "#ECEFF1"     # Pale Gray-Blue (Road network & Geography)
C_HIST = "#FFEBEE"     # Pale Pink/Red (Outbreak history & lags)

C_ENC = "#D6E4F0"      # Muted Blue (GraphSAGE Encoder)
C_ENC_BORDER = "#4A90E2"
C_VEC = "#E8EAF6"      # Soft Indigo (Feature vectors)
C_EMB = "#BBDEFB"      # Soft Blue (Node embedding)
C_HEAD = "#E0F2F1"     # Pale Teal (Classification head)
C_HEAD_BORDER = "#26A69A"
C_POS = "#C8E6C9"      # Soft Green (Positive risk)
C_NEG = "#FFCDD2"      # Soft Red (Negative background)
C_PANEL_BG = "#FAFAFA" # Sub-panel background
C_BORDER = "#455A64"   # Main element outline color
C_TEXT = "#263238"     # Main dark text color
C_MUTED = "#607D8B"    # Muted secondary text
C_ACCENT = "#1976D2"   # Primary highlight accent

plt.rcParams['font.sans-serif'] = 'DejaVu Sans', 'Arial', 'Helvetica', 'Liberation Sans'
plt.rcParams['font.family'] = 'sans-serif'

def draw_rounded_box(ax, x, y, w, h, bg_color, border_color=C_BORDER, lw=1.0, pad=0.8, zorder=2, linestyle='-'):
    """Draws a clean rounded rectangle using FancyBboxPatch."""
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle=f"round,pad={pad},rounding_size=1.2",
                         facecolor=bg_color,
                         edgecolor=border_color,
                         linewidth=lw,
                         linestyle=linestyle,
                         zorder=zorder)
    ax.add_patch(box)
    return box

def draw_arrow(ax, x1, y1, x2, y2, color="#546E7A", lw=1.2, arrowstyle="-|>", mutation_scale=12, zorder=4):
    """Draws a simple, elegant directional arrow."""
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=arrowstyle,
                                color=color,
                                lw=lw,
                                mutation_scale=mutation_scale,
                                shrinkA=0, shrinkB=0),
                zorder=zorder)

def build_figure():
    fig, ax = plt.subplots(figsize=(19, 13.5), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    
    # -------------------------------------------------------------
    # PANEL (a): COMPLETE PREDICTION PIPELINE (Top Half: y=47.5 to 98.5)
    # -------------------------------------------------------------
    draw_rounded_box(ax, 1.5, 47.5, 97.0, 51.0, "#FFFFFF", border_color="#B0BEC5", lw=1.1, pad=0.5, zorder=1)
    ax.text(3.0, 97.2, "(a) Complete HPAI Spatial GraphSAGE Prediction Pipeline", 
            fontsize=12, fontweight='bold', color=C_TEXT, va='top')
    
    # 1. Six vertically stacked input data blocks
    inputs = [
        ("Land Cover & Vegetation", "Dynamic World: 8 class fractions\n(cropland, water, vegetation, built)", C_ENV, 88.0),
        ("Climate & Meteorology", "ERA5-Land: 2m temp, precip,\nanomalies (mean, max, std)", C_CLIM, 80.8),
        ("Host Poultry Population", "GLW + Census: chicken & duck\ndensities (mean, max, p95)", C_HOST, 73.6),
        ("Wild-Bird Surveillance", "eBird & CFIA detections,\nspecies-group indicators", C_BIRD, 66.4),
        ("Roads & Geography", "Road length, highway distance,\nCSD area, connectivity", C_ROAD, 59.2),
        ("Surveillance Outbreak Lags", "1-, 2-, 3-, 6-, 12-mo CSD &\nprovincial outbreak counts", C_HIST, 52.0),
    ]
    
    for title, desc, col, y_pos in inputs:
        draw_rounded_box(ax, 3.2, y_pos, 17.8, 5.8, col, border_color="#78909C", lw=0.9)
        ax.text(4.0, y_pos + 4.2, title, fontsize=8.2, fontweight='bold', color=C_TEXT)
        ax.text(4.0, y_pos + 1.1, desc, fontsize=6.8, color="#37474F", linespacing=1.1)
        # Clean direct arrow to feature constructor
        draw_arrow(ax, 21.3, y_pos + 2.9, 24.8, y_pos + 2.9, color="#90A4AE", lw=1.0)
    
    # 2. CSD Feature Construction Block
    draw_rounded_box(ax, 25.2, 51.5, 11.8, 42.5, C_VEC, border_color="#5C6BC0", lw=1.1)
    ax.text(31.1, 92.5, "Monthly CSD\nFeature Vector", fontsize=8.5, fontweight='bold', color="#283593", ha='center', linespacing=1.1)
    ax.text(31.1, 88.0, r"$\mathbf{x}_{i,t} \in \mathbb{R}^{62}$", fontsize=9.5, fontweight='bold', color=C_ACCENT, ha='center')
    
    # 43 non-lag vs 19 lag division inside vector
    draw_rounded_box(ax, 26.0, 68.5, 10.2, 17.5, "#FFFFFF", border_color="#9FA8DA", lw=0.8)
    ax.text(31.1, 83.5, "43 Non-Lag\nPredictors", fontsize=7.8, fontweight='bold', color="#3949AB", ha='center', linespacing=1.1)
    ax.text(31.1, 75.0, "Static & Environmental\nLandscape Context\n(Land, Climate,\nHosts, Roads)", fontsize=6.5, color="#5C6BC0", ha='center', linespacing=1.15)
    
    draw_rounded_box(ax, 26.0, 53.0, 10.2, 14.0, "#FFFFFF", border_color="#EF9A9A", lw=0.8)
    ax.text(31.1, 64.5, "19 History Lags", fontsize=7.8, fontweight='bold', color="#C62828", ha='center')
    ax.text(31.1, 57.5, "Outbreak History &\nRegional Pressure\n(1 to 12 Months)", fontsize=6.5, color="#D32F2F", ha='center', linespacing=1.15)
    
    # Arrow to GraphSAGE Encoder
    draw_arrow(ax, 37.3, 72.5, 40.8, 72.5, color=C_TEXT, lw=1.3)
    
    # 3. 2-Layer GraphSAGE Encoder Block (Large Muted Blue Box)
    draw_rounded_box(ax, 41.2, 51.5, 26.5, 42.5, C_ENC, border_color=C_ENC_BORDER, lw=1.3)
    ax.text(54.45, 92.5, "Spatial GraphSAGE Encoder (2 Layers)", fontsize=9.5, fontweight='bold', color="#1565C0", ha='center')
    ax.text(54.45, 89.2, r"Inductive Neighborhood Aggregation over $\mathcal{G} = (\mathcal{V}, \mathcal{E})$", fontsize=7.5, color="#1976D2", ha='center')
    
    # Inside GraphSAGE Encoder: Diagram of 2 layers
    # Layer 1 box
    draw_rounded_box(ax, 42.6, 71.0, 11.0, 16.0, "#FFFFFF", border_color="#90CAF9", lw=0.9)
    ax.text(48.1, 84.5, "GraphSAGE Layer 1", fontsize=7.8, fontweight='bold', color="#0D47A1", ha='center')
    ax.text(48.1, 80.5, r"$\mathbf{h}_i^{(1)} = \sigma(W_1 [\mathbf{x}_i \,\|\, \bar{\mathbf{x}}_{\mathcal{N}(i)}])$", fontsize=6.8, color=C_TEXT, ha='center')
    ax.text(48.1, 76.5, "Dim: 62 → 64", fontsize=7.2, color=C_MUTED, ha='center')
    ax.text(48.1, 73.0, "ReLU + Dropout (0.2)", fontsize=6.8, color="#546E7A", ha='center')
    
    # Arrow between Layer 1 and Layer 2
    draw_arrow(ax, 54.0, 79.0, 56.5, 79.0, color="#1976D2", lw=1.1)
    
    # Layer 2 box
    draw_rounded_box(ax, 57.0, 71.0, 9.5, 16.0, "#FFFFFF", border_color="#90CAF9", lw=0.9)
    ax.text(61.75, 84.5, "GraphSAGE Layer 2", fontsize=7.8, fontweight='bold', color="#0D47A1", ha='center')
    ax.text(61.75, 80.5, r"$\mathbf{z}_i = \sigma(W_2 [\mathbf{h}_i^{(1)} \,\|\, \bar{\mathbf{h}}_{\mathcal{N}(i)}^{(1)}])$", fontsize=6.8, color=C_TEXT, ha='center')
    ax.text(61.75, 76.5, "Dim: 64 → 64", fontsize=7.2, color=C_MUTED, ha='center')
    ax.text(61.75, 73.0, "ReLU Activation", fontsize=6.8, color="#546E7A", ha='center')
    
    # Bottom note in Encoder
    draw_rounded_box(ax, 42.6, 53.0, 23.9, 15.5, "#F5F9FD", border_color="#BBDEFB", lw=0.8)
    ax.text(54.45, 65.5, r"Fixed Spatial Graph: Centroid $k\text{NN}-3$ Topology", fontsize=7.8, fontweight='bold', color="#1565C0", ha='center')
    ax.text(54.45, 61.0, "5,054 CSD Nodes  |  9,655 Unique Undirected Edges", fontsize=7.0, fontweight='bold', color="#37474F", ha='center')
    ax.text(54.45, 56.5, "• Dynamic node features per month t\n• Mean aggregation over 3 nearest geographic neighbors\n• Bidirectional spatial information propagation", fontsize=6.5, color="#455A64", ha='center', linespacing=1.2)
    
    # Arrow to Node Embedding
    draw_arrow(ax, 68.0, 72.5, 71.0, 72.5, color=C_TEXT, lw=1.3)
    
    # 4. Learned Node Embedding Block
    draw_rounded_box(ax, 71.5, 62.0, 6.2, 21.0, C_EMB, border_color="#1E88E5", lw=1.1)
    ax.text(74.6, 79.5, "Node\nEmbedding", fontsize=8.2, fontweight='bold', color="#0D47A1", ha='center', linespacing=1.1)
    ax.text(74.6, 72.5, r"$\mathbf{z}_{i,t} \in \mathbb{R}^{64}$", fontsize=9.5, fontweight='bold', color=C_ACCENT, ha='center')
    ax.text(74.6, 65.5, "Aggregated\nSpatial Context", fontsize=6.8, color="#1565C0", ha='center', linespacing=1.1)
    
    # Arrow to Classification Head
    draw_arrow(ax, 78.0, 72.5, 81.0, 72.5, color=C_TEXT, lw=1.3)
    
    # 5. Classification Head Block
    draw_rounded_box(ax, 81.5, 62.0, 8.2, 21.0, C_HEAD, border_color=C_HEAD_BORDER, lw=1.1)
    ax.text(85.6, 79.5, "Classification\nHead", fontsize=8.2, fontweight='bold', color="#00695C", ha='center', linespacing=1.1)
    ax.text(85.6, 73.0, "Linear Layer\n(64 → 1)", fontsize=7.5, color="#004D40", ha='center', linespacing=1.1)
    ax.text(85.6, 66.0, r"Sigmoid $\sigma(\cdot)$" + "\nActivation", fontsize=7.5, color="#004D40", ha='center', linespacing=1.1)
    
    # Arrow to Output
    draw_arrow(ax, 90.0, 72.5, 92.2, 72.5, color=C_TEXT, lw=1.3)
    
    # 6. Output Risk Probability
    draw_rounded_box(ax, 92.5, 62.0, 5.2, 21.0, "#FFFFFF", border_color="#43A047", lw=1.2)
    ax.text(95.1, 79.5, "Outbreak\nRisk", fontsize=8.2, fontweight='bold', color="#1B5E20", ha='center', linespacing=1.1)
    ax.text(95.1, 73.0, r"$\hat{p}_{i,t+1}$", fontsize=10.5, fontweight='bold', color="#2E7D32", ha='center')
    ax.text(95.1, 68.5, r"$P(Y_{i,t+1}=1)$", fontsize=7.0, color="#388E3C", ha='center')
    
    # Small visual risk bar icon
    draw_rounded_box(ax, 93.3, 64.0, 3.6, 3.0, C_POS, border_color="#66BB6A", lw=0.6, pad=0.2)
    ax.text(95.1, 65.5, "Risk Score", fontsize=5.8, fontweight='bold', color="#1B5E20", ha='center', va='center')
    
    # Bottom Annotation inside Panel a
    ax.text(50.0, 48.8, "Spatial Prediction Unit: Canadian Census Subdivision (CSD) | Target: One-month-ahead binary outbreak occurrence | Evaluated on Locked 2025 Test Set", 
            fontsize=7.2, color=C_MUTED, ha='center', style='italic')
    
    
    # -------------------------------------------------------------
    # BOTTOM ROW PANELS (y=2 to 45)
    # -------------------------------------------------------------
    
    # -------------------------------------------------------------
    # PANEL (b): MONTHLY CSD FEATURE REPRESENTATION (x=1.5 to 24.5)
    # -------------------------------------------------------------
    draw_rounded_box(ax, 1.5, 2.0, 23.0, 43.5, "#FFFFFF", border_color="#B0BEC5", lw=1.0, pad=0.5, zorder=1)
    ax.text(2.5, 43.8, "(b) Monthly CSD Feature Vector", fontsize=10.0, fontweight='bold', color=C_TEXT)
    
    # CSD Node representation at top of Panel b
    draw_rounded_box(ax, 2.8, 36.5, 20.4, 5.2, "#ECEFF1", border_color="#78909C", lw=0.8)
    ax.text(13.0, 39.1, r"CSD $i \in \{1, \dots, 5054\}$, Month $t \in \{1, \dots, 47\}$", fontsize=7.5, fontweight='bold', color=C_TEXT, ha='center')
    
    # 43 Non-lag bracket
    draw_rounded_box(ax, 2.8, 20.5, 20.4, 14.5, C_VEC, border_color="#9FA8DA", lw=0.9)
    ax.text(3.6, 33.2, "43 Non-Lag Features (Static & Landscape)", fontsize=7.5, fontweight='bold', color="#283593")
    
    non_lag_items = [
        ("• Dynamic World Land Cover:", "8 class fractions (crops, water, etc.)"),
        ("• ERA5-Land Climate Data:", "2m temperature, total precipitation"),
        ("• Poultry Host Populations:", "GLW chicken & duck (mean, max, p95)"),
        ("• Transport Infrastructure:", "Major road length, highway distance"),
        ("• Physical Geography:", "CSD polygon area, elevation"),
    ]
    for idx, (head, body) in enumerate(non_lag_items):
        y_item = 30.5 - idx * 2.3
        ax.text(3.8, y_item, f"{head} {body}", fontsize=6.5, color="#37474F")
        
    # 19 History/Lag bracket
    draw_rounded_box(ax, 2.8, 3.5, 20.4, 15.5, "#FFEBEE", border_color="#EF9A9A", lw=0.9)
    ax.text(3.6, 17.2, "19 Outbreak-History & Surveillance Lags", fontsize=7.5, fontweight='bold', color="#C62828")
    
    lag_items = [
        ("• Same-CSD Outbreak History:", "Lags over 1, 2, 3, 6, 12 months"),
        ("• Provincial Outbreak Pressure:", "Lags over 1, 2, 3, 6, 12 months"),
        ("• Wild-Bird HPAI Surveillance:", "Regional detections (1, 3, 6 months)"),
        ("• Neighbor Outbreak Pressure:", "Weighted adjacent historical activity"),
    ]
    for idx, (head, body) in enumerate(lag_items):
        y_item = 14.5 - idx * 2.3
        ax.text(3.8, y_item, f"{head} {body}", fontsize=6.5, color="#37474F")
        
    ax.text(13.0, 4.2, "Strict temporal cutoff: values observed strictly on or before month t", 
            fontsize=6.0, color="#C62828", ha='center', style='italic')

    # -------------------------------------------------------------
    # PANEL (c): GRAPHSAGE MESSAGE PASSING (x=26.0 to 51.5)
    # -------------------------------------------------------------
    draw_rounded_box(ax, 26.0, 2.0, 25.5, 43.5, "#FFFFFF", border_color="#B0BEC5", lw=1.0, pad=0.5, zorder=1)
    ax.text(27.0, 43.8, "(c) Spatial GraphSAGE Message Passing", fontsize=10.0, fontweight='bold', color=C_TEXT)
    
    # Central Node i and Neighbors j1, j2, j3
    node_i = (32.0, 33.0)
    node_j1 = (28.5, 39.5)
    node_j2 = (35.5, 39.5)
    node_j3 = (28.5, 26.5)
    
    # Connecting edges
    ax.plot([node_i[0], node_j1[0]], [node_i[1], node_j1[1]], color="#90A4AE", lw=1.5, zorder=2)
    ax.plot([node_i[0], node_j2[0]], [node_i[1], node_j2[1]], color="#90A4AE", lw=1.5, zorder=2)
    ax.plot([node_i[0], node_j3[0]], [node_i[1], node_j3[1]], color="#90A4AE", lw=1.5, zorder=2)
    
    # Draw neighbor nodes
    for pt, lbl in [(node_j1, r"$j_1$"), (node_j2, r"$j_2$"), (node_j3, r"$j_3$")]:
        c = plt.Circle(pt, 1.3, facecolor="#E3F2FD", edgecolor="#1976D2", lw=1.0, zorder=3)
        ax.add_patch(c)
        ax.text(pt[0], pt[1], lbl, fontsize=7.5, fontweight='bold', color="#0D47A1", ha='center', va='center', zorder=4)
        
    # Central node i
    c_i = plt.Circle(node_i, 1.5, facecolor="#FFE082", edgecolor="#F57C00", lw=1.3, zorder=3)
    ax.add_patch(c_i)
    ax.text(node_i[0], node_i[1], r"$i$", fontsize=8.5, fontweight='bold', color="#E65100", ha='center', va='center', zorder=4)
    
    # Message arrows from neighbors into Mean Aggregator
    draw_rounded_box(ax, 39.5, 32.5, 10.5, 9.0, "#FFF3E0", border_color="#FFB74D", lw=1.0)
    ax.text(44.75, 39.2, "Neighborhood\nMean Aggregation", fontsize=7.0, fontweight='bold', color="#E65100", ha='center', linespacing=1.1)
    ax.text(44.75, 34.5, r"$\bar{\mathbf{h}}_{\mathcal{N}(i)} = \frac{1}{|\mathcal{N}(i)|}\sum_{j \in \mathcal{N}(i)} \mathbf{h}_j$", fontsize=6.8, color=C_TEXT, ha='center')
    
    draw_arrow(ax, 33.5, 34.5, 39.0, 35.5, color="#FB8C00", lw=1.1)
    
    # Concatenation and Transformation Block
    draw_rounded_box(ax, 28.0, 17.5, 22.0, 7.5, "#E8F5E9", border_color="#81C784", lw=1.0)
    ax.text(39.0, 22.8, r"Concatenate Self + Neighborhood Context", fontsize=7.2, fontweight='bold', color="#1B5E20", ha='center')
    ax.text(39.0, 19.5, r"$\mathbf{h}_i^{(l+1)} = \sigma \left( W^{(l)} \left[ \mathbf{h}_i^{(l)} \,\|\, \bar{\mathbf{h}}_{\mathcal{N}(i)}^{(l)} \right] \right)$", 
            fontsize=8.0, fontweight='bold', color="#2E7D32", ha='center')
    
    draw_arrow(ax, 44.75, 32.0, 44.75, 25.5, color="#2E7D32", lw=1.1)
    
    # Output Node Representation
    draw_rounded_box(ax, 34.0, 10.0, 10.0, 5.0, C_EMB, border_color="#1E88E5", lw=1.0)
    ax.text(39.0, 12.5, r"Updated $\mathbf{h}_i^{(l+1)}$ or Output $\mathbf{z}_i$", fontsize=7.2, fontweight='bold', color="#0D47A1", ha='center', va='center')
    
    draw_arrow(ax, 39.0, 17.0, 39.0, 15.5, color="#1976D2", lw=1.1)
    
    # Scientific note on topology
    ax.text(38.75, 6.2, "• Symmetrized kNN-3 Centroid Proximity Graph\n• Edges represent geographic proximity, not known transmission routes", 
            fontsize=6.2, color="#546E7A", ha='center', linespacing=1.2)

    # -------------------------------------------------------------
    # PANEL (d): FORWARD-TIME PREDICTION (x=53.0 to 76.5)
    # -------------------------------------------------------------
    draw_rounded_box(ax, 53.0, 2.0, 23.5, 43.5, "#FFFFFF", border_color="#B0BEC5", lw=1.0, pad=0.5, zorder=1)
    ax.text(54.0, 43.8, "(d) Forward-Time Temporal Evaluation", fontsize=10.0, fontweight='bold', color=C_TEXT)
    
    # Month timeline: t-2 -> t-1 -> t -> t+1
    draw_rounded_box(ax, 54.5, 34.5, 20.5, 7.5, "#ECEFF1", border_color="#90A4AE", lw=0.9)
    ax.text(64.75, 40.0, "Monthly Rolling Prediction Setup", fontsize=7.5, fontweight='bold', color="#37474F", ha='center')
    
    months = [r"$t-2$", r"$t-1$", r"$t$"]
    for i, m_lbl in enumerate(months):
        bx = 55.5 + i * 3.8
        draw_rounded_box(ax, bx, 35.5, 3.2, 3.2, "#FFFFFF", border_color="#78909C", lw=0.7, pad=0.1)
        ax.text(bx + 1.6, 37.1, m_lbl, fontsize=7.0, color=C_TEXT, ha='center', va='center')
        if i < 2:
            draw_arrow(ax, bx + 3.3, 37.1, bx + 3.7, 37.1, color="#78909C", lw=0.9)
            
    # Red Barrier: Leakage prevention
    ax.plot([67.5, 67.5], [35.0, 41.5], color="#D32F2F", lw=1.8, linestyle='--', zorder=4)
    ax.text(67.5, 34.7, "Strict\nCutoff", fontsize=5.8, fontweight='bold', color="#D32F2F", ha='center', va='top')
    
    # Predict t+1 box
    draw_rounded_box(ax, 69.0, 35.5, 4.8, 3.2, C_POS, border_color="#43A047", lw=1.0, pad=0.1)
    ax.text(71.4, 37.1, r"Target $t+1$", fontsize=7.0, fontweight='bold', color="#1B5E20", ha='center', va='center')
    
    # Experimental split boxes
    splits = [
        ("Training Partition", "2022–2023 (24 Months)", "121,296 CSD-months  |  262 Outbreaks", "#E8EAF6", "#3F51B5", 25.5),
        ("Validation Partition", "2024 (12 Months)", "60,648 CSD-months  |  40 Outbreaks", "#FFF8E1", "#FFA000", 17.5),
        ("Locked Test Partition", "2025 (11 Months)", "55,594 CSD-months  |  79 Outbreaks", "#E8F5E9", "#43A047", 9.5),
    ]
    
    for title, span, detail, bg, border, y_sp in splits:
        draw_rounded_box(ax, 54.5, y_sp, 20.5, 6.5, bg, border_color=border, lw=1.0)
        ax.text(55.5, y_sp + 4.8, title, fontsize=7.5, fontweight='bold', color=border)
        ax.text(55.5, y_sp + 2.8, span, fontsize=6.8, fontweight='bold', color="#263238")
        ax.text(55.5, y_sp + 1.0, detail, fontsize=6.2, color="#455A64")
        
    ax.text(64.75, 4.0, "• Feature scalers/imputers fit strictly on 2022–2023 Train\n• Model weights permanently locked before 2025 Test evaluation", 
            fontsize=6.0, color="#37474F", ha='center', linespacing=1.2)

    # -------------------------------------------------------------
    # PANEL (e): CORE SCIENTIFIC COMPARISON (x=78.0 to 98.5)
    # -------------------------------------------------------------
    draw_rounded_box(ax, 78.0, 2.0, 20.5, 43.5, "#FFFFFF", border_color="#B0BEC5", lw=1.0, pad=0.5, zorder=1)
    ax.text(79.0, 43.8, "(e) Core Spatial Message-Passing Lift", fontsize=10.0, fontweight='bold', color=C_TEXT)
    
    # Full Feature Comparison (62 features)
    draw_rounded_box(ax, 79.2, 23.5, 18.0, 18.5, "#F5F5F5", border_color="#BDBDBD", lw=0.9)
    ax.text(88.2, 40.0, "Full Model (62 Predictors)", fontsize=7.8, fontweight='bold', color="#212121", ha='center')
    
    # GraphSAGE vs MLP mini boxes
    draw_rounded_box(ax, 80.2, 33.5, 7.5, 5.0, C_ENC, border_color="#1976D2", lw=1.0)
    ax.text(83.95, 36.8, "GraphSAGE", fontsize=7.2, fontweight='bold', color="#0D47A1", ha='center')
    ax.text(83.95, 34.5, "AP = 0.1107", fontsize=7.5, fontweight='bold', color="#1565C0", ha='center')
    
    draw_rounded_box(ax, 88.7, 33.5, 7.5, 5.0, "#ECEFF1", border_color="#78909C", lw=1.0)
    ax.text(92.45, 36.8, "Matched MLP", fontsize=7.2, fontweight='bold', color="#37474F", ha='center')
    ax.text(92.45, 34.5, "AP = 0.0704", fontsize=7.5, fontweight='bold', color="#455A64", ha='center')
    
    # Delta AP box
    draw_rounded_box(ax, 80.2, 24.5, 16.0, 7.5, "#E8F5E9", border_color="#66BB6A", lw=0.8)
    ax.text(88.2, 29.8, r"$\Delta\mathbf{AP} = \mathbf{+0.0350}$ (95% CI $[+0.0040, +0.0720]$)", fontsize=6.8, fontweight='bold', color="#1B5E20", ha='center')
    ax.text(88.2, 27.2, "Bootstrap Win Rate: 98.7% (Significant)", fontsize=6.2, color="#2E7D32", ha='center')
    ax.text(88.2, 25.2, "77.9× lift over random prevalence (0.00142)", fontsize=5.8, color="#388E3C", ha='center')
    
    # Strict No-Lag Ablation (43 features)
    draw_rounded_box(ax, 79.2, 3.5, 18.0, 18.5, "#F5F5F5", border_color="#BDBDBD", lw=0.9)
    ax.text(88.2, 20.0, "No-Lag Ablation (43 Predictors)", fontsize=7.8, fontweight='bold', color="#212121", ha='center')
    
    draw_rounded_box(ax, 80.2, 13.5, 7.5, 5.0, C_ENC, border_color="#1976D2", lw=1.0)
    ax.text(83.95, 16.8, "GraphSAGE", fontsize=7.2, fontweight='bold', color="#0D47A1", ha='center')
    ax.text(83.95, 14.5, "AP = 0.0760", fontsize=7.5, fontweight='bold', color="#1565C0", ha='center')
    
    draw_rounded_box(ax, 88.7, 13.5, 7.5, 5.0, "#ECEFF1", border_color="#78909C", lw=1.0)
    ax.text(92.45, 16.8, "Matched MLP", fontsize=7.2, fontweight='bold', color="#37474F", ha='center')
    ax.text(92.45, 14.5, "AP = 0.0462", fontsize=7.5, fontweight='bold', color="#455A64", ha='center')
    
    # Delta AP box
    draw_rounded_box(ax, 80.2, 4.5, 16.0, 7.5, "#E8F5E9", border_color="#66BB6A", lw=0.8)
    ax.text(88.2, 9.8, r"$\Delta\mathbf{AP} = \mathbf{+0.0320}$ (95% CI $[+0.0060, +0.0640]$)", fontsize=6.8, fontweight='bold', color="#1B5E20", ha='center')
    ax.text(88.2, 7.2, "Bootstrap Win Rate: 100.0% (Significant)", fontsize=6.2, color="#2E7D32", ha='center')
    ax.text(88.2, 5.2, "Message passing extracts independent landscape signal", fontsize=5.6, color="#388E3C", ha='center')
    
    
    # -------------------------------------------------------------
    # DOTTED EXPANSION CONNECTORS (Subtle Guide lines)
    # -------------------------------------------------------------
    # From Feature Vector in (a) to Panel (b)
    ax.plot([31.1, 13.0], [51.0, 45.8], color="#9FA8DA", linestyle=":", lw=1.2, zorder=2)
    
    # From GraphSAGE Encoder in (a) to Panel (c)
    ax.plot([54.45, 38.75], [51.0, 45.8], color="#90CAF9", linestyle=":", lw=1.2, zorder=2)
    
    plt.tight_layout()
    
    # Save outputs in PDF, SVG, and high-res PNG formats
    out_base_latex = OUT_DIR_LATEX / "Figure_architecture_HPAI_GraphSAGE"
    out_base_final = OUT_DIR_FINAL / "Figure1_schematic"
    
    for fmt in ['pdf', 'svg', 'png']:
        f_lat = out_base_latex.with_suffix(f".{fmt}")
        fig.savefig(f_lat, bbox_inches='tight', dpi=300, format=fmt)
        print(f"Saved: {f_lat}")
        
        f_fin = out_base_final.with_suffix(f".{fmt}")
        fig.savefig(f_fin, bbox_inches='tight', dpi=300, format=fmt)
        print(f"Saved: {f_fin}")

    plt.close(fig)
    print("All architecture figure formats successfully exported.")

if __name__ == "__main__":
    build_figure()
