"""
Redesign Figure 1: Clean Scientific GNN Architecture for Elsevier Manuscript.

Final Technical-Accuracy and Simplification Pass:
1. Removed top "TEMPORAL PIPELINE" banner and duplicate equation.
2. Section headings simplified to:
   - "Data Representation"
   - "Spatial Graph Learning"
   - "Risk Forecasting"
3. Six input data blocks with miniature scientific figures and concise labels:
   - Land Cover, Climate, Poultry, Wild Birds, Road Network, Surveillance (no sub-labels).
4. Feature matrix X_t in R^(5054 x 62) with highlighted x_{i,t} row retained.
5. 14-node spatial kNN-3 graph with highlighted target i and neighbors j1, j2, j3 retained.
6. Graph text simplified to:
   - Spatial Graph G=(V,E)
   - kNN, k=3
   - Centroid proximity
7. GraphSAGE message passing explicitly shows:
   - Inset explaining the layer-wise operation:
     Neighbor features -> Mean aggregation -> Concatenate with self -> Linear + activation
   - Macro architecture: GraphSAGE Layer 1 -> GraphSAGE Layer 2 -> Spatial node embedding z_{i,t}
   - One compact GraphSAGE equation retained.
8. Embedding vector z_{i,t} in R^64 retained.
9. Risk Forecasting simplified:
   - Example CSD probability numbers table removed completely.
   - Shows: z_{i,t} -> Linear Layer -> Sigmoid -> p̂_{i,t+1} -> HPAI Outbreak Risk (Month t+1).
10. Subtle temporal annotation at the bottom: "Features at month t → Risk at month t+1".
11. Technical GNN paper visual language, slightly less dense, clean white background.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, Circle

# Matplotlib configuration for publication quality
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']

def draw_tiny_raster(ax, x, y, w, h):
    """Miniature land cover raster grid (4x4 pixels)."""
    grid = [
        ['#10b981', '#10b981', '#34d399', '#3b82f6'],
        ['#10b981', '#34d399', '#f59e0b', '#3b82f6'],
        ['#34d399', '#f59e0b', '#f59e0b', '#3b82f6'],
        ['#059669', '#34d399', '#f59e0b', '#1d4ed8']
    ]
    dx, dy = w / 4, h / 4
    for r in range(4):
        for c in range(4):
            ax.add_patch(Rectangle((x + c*dx, y + (3-r)*dy), dx, dy,
                                   facecolor=grid[r][c], edgecolor='#ffffff', lw=0.4, zorder=4))
    ax.add_patch(Rectangle((x, y), w, h, facecolor='none', edgecolor='#64748b', lw=0.6, zorder=5))

def draw_tiny_climate(ax, x, y, w, h):
    """Miniature ERA5 climate curve (temperature line + precipitation bars)."""
    ax.add_patch(Rectangle((x, y), w, h, facecolor='#f8fafc', edgecolor='#64748b', lw=0.6, zorder=3))
    bw = w * 0.12
    for bx, bh in zip(np.linspace(x + w*0.15, x + w*0.85, 4), [0.45, 0.70, 0.35, 0.60]):
        ax.add_patch(Rectangle((bx - bw/2, y), bw, h*bh*0.65, facecolor='#93c5fd', edgecolor='none', zorder=4))
    xs = np.linspace(x + w*0.06, x + w*0.94, 15)
    yt = y + h*0.52 + h*0.32 * np.sin(np.linspace(0, 2.2*np.pi, 15))
    ax.plot(xs, yt, color='#0284c7', lw=1.1, zorder=5)

def draw_tiny_density(ax, x, y, w, h):
    """Miniature poultry density heatmap (3x3 grid)."""
    cells = [
        ['#fff7ed', '#fed7aa', '#fb923c'],
        ['#fed7aa', '#fb923c', '#ea580c'],
        ['#fb923c', '#ea580c', '#9a3412']
    ]
    dx, dy = w / 3, h / 3
    for r in range(3):
        for c in range(3):
            ax.add_patch(Rectangle((x + c*dx, y + (2-r)*dy), dx, dy,
                                   facecolor=cells[r][c], edgecolor='#ffffff', lw=0.4, zorder=4))
    ax.add_patch(Rectangle((x, y), w, h, facecolor='none', edgecolor='#64748b', lw=0.6, zorder=5))

def draw_tiny_wild_birds(ax, x, y, w, h):
    """Miniature wild-bird temporal abundance curve with seasonal migration peak."""
    ax.add_patch(Rectangle((x, y), w, h, facecolor='#f8fafc', edgecolor='#64748b', lw=0.6, zorder=3))
    xs = np.linspace(x + w*0.06, x + w*0.94, 20)
    t = np.linspace(0, 1, 20)
    curve = 0.85 * np.exp(-((t - 0.35)**2) / 0.02) + 0.95 * np.exp(-((t - 0.78)**2) / 0.02) + 0.12
    yt = y + h * 0.12 + h * 0.70 * (curve / np.max(curve))
    ax.fill_between(xs, y, yt, color='#ede9fe', alpha=0.8, zorder=4)
    ax.plot(xs, yt, color='#6366f1', lw=1.1, zorder=5)

def draw_tiny_roads(ax, x, y, w, h):
    """Miniature road network topology (4 nodes and intersecting segments)."""
    ax.add_patch(Rectangle((x, y), w, h, facecolor='#f8fafc', edgecolor='#64748b', lw=0.6, zorder=3))
    pts = [(x + w*0.15, y + h*0.25), (x + w*0.85, y + h*0.80),
           (x + w*0.20, y + h*0.85), (x + w*0.80, y + h*0.20)]
    ax.plot([pts[0][0], pts[1][0]], [pts[0][1], pts[1][1]], color='#475569', lw=1.5, zorder=4)
    ax.plot([pts[2][0], pts[3][0]], [pts[2][1], pts[3][1]], color='#64748b', lw=1.1, zorder=4)
    for px, py in pts:
        ax.add_patch(Circle((px, py), w*0.07, facecolor='#334155', edgecolor='white', lw=0.4, zorder=5))

def draw_tiny_surveillance(ax, x, y, w, h):
    """Miniature surveillance outbreak history matrix (discrete lag bars)."""
    ax.add_patch(Rectangle((x, y), w, h, facecolor='#f8fafc', edgecolor='#64748b', lw=0.6, zorder=3))
    cols = ['#ef4444', '#ef4444', '#f87171', '#cbd5e1', '#ef4444', '#cbd5e1', '#cbd5e1']
    bw = w / len(cols)
    for i, c in enumerate(cols):
        bh = h * (0.85 if c != '#cbd5e1' else 0.25)
        ax.add_patch(Rectangle((x + i*bw + bw*0.1, y), bw*0.8, bh, facecolor=c, edgecolor='none', zorder=4))
    ax.add_patch(Rectangle((x, y), w, h, facecolor='none', edgecolor='#64748b', lw=0.6, zorder=5))

def generate_figure():
    # Dimensions: 16.5 inches wide, 5.7 inches high (less dense, spacious academic layout)
    fig = plt.figure(figsize=(16.5, 5.7), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 16.5)
    ax.set_ylim(0, 5.7)
    ax.axis('off')

    # Canvas Background
    ax.add_patch(Rectangle((0, 0), 16.5, 5.7, facecolor='#ffffff', edgecolor='none', zorder=0))

    # Palettes
    c_title = "#0f172a"
    c_body = "#334155"
    c_muted = "#64748b"
    c_sec_border = "#cbd5e1"
    c_sec_bg = "#ffffff"

    # =========================================================================
    # THREE MAJOR GROUPED SECTIONS (Enclosing boundaries)
    # =========================================================================
    y_sec = 0.36
    h_sec = 5.02

    sec1_x, sec1_w = 0.45, 4.15   # Data Representation
    sec2_x, sec2_w = 4.85, 8.45   # Spatial Graph Learning (Main Focus)
    sec3_x, sec3_w = 13.55, 2.50  # Risk Forecasting

    sections = [
        (sec1_x, sec1_w, "Data Representation", "#0284c7"),
        (sec2_x, sec2_w, "Spatial Graph Learning", "#2563eb"),
        (sec3_x, sec3_w, "Risk Forecasting", "#b91c1c")
    ]

    for sx, sw, stitle, scol in sections:
        ax.add_patch(Rectangle((sx, y_sec), sw, h_sec, facecolor=c_sec_bg, edgecolor=c_sec_border, lw=0.9, ls='--', zorder=1))
        # Simplified clean section badge
        tag_w = len(stitle) * 0.082 + 0.36
        ax.add_patch(Rectangle((sx + 0.15, y_sec + h_sec - 0.28), tag_w, 0.28, facecolor=scol, edgecolor='none', zorder=3))
        ax.text(sx + 0.15 + tag_w/2, y_sec + h_sec - 0.14, stitle, ha='center', va='center', fontsize=8.0, fontweight='bold', color='white', zorder=4)

    # =========================================================================
    # SECTION 1: DATA REPRESENTATION
    # =========================================================================
    # 6 stacked input blocks with concise single labels (no sub-labels)
    domains = [
        ("Land Cover", "raster", "#10b981"),
        ("Climate", "climate", "#0284c7"),
        ("Poultry", "density", "#ea580c"),
        ("Wild Birds", "wild", "#6366f1"),
        ("Road Network", "roads", "#475569"),
        ("Surveillance", "surv", "#dc2626")
    ]

    b_x = sec1_x + 0.18
    b_w = 1.55
    b_h = 0.52
    b_y_top = y_sec + h_sec - 0.55
    b_dy = 0.68

    for idx, (dname, dtype, dcol) in enumerate(domains):
        by = b_y_top - idx * b_dy - b_h
        ax.add_patch(Rectangle((b_x, by), b_w, b_h, facecolor='#ffffff', edgecolor='#cbd5e1', lw=0.8, zorder=2))
        ax.add_patch(Rectangle((b_x, by), 0.06, b_h, facecolor=dcol, edgecolor='none', zorder=3))
        
        gx = b_x + 0.12
        gy = by + (b_h - 0.34) / 2
        gw = 0.34
        gh = 0.34
        if dtype == "raster":
            draw_tiny_raster(ax, gx, gy, gw, gh)
        elif dtype == "climate":
            draw_tiny_climate(ax, gx, gy, gw, gh)
        elif dtype == "density":
            draw_tiny_density(ax, gx, gy, gw, gh)
        elif dtype == "wild":
            draw_tiny_wild_birds(ax, gx, gy, gw, gh)
        elif dtype == "roads":
            draw_tiny_roads(ax, gx, gy, gw, gh)
        elif dtype == "surv":
            draw_tiny_surveillance(ax, gx, gy, gw, gh)

        # Single clear bold label centered vertically
        ax.text(b_x + 0.54, by + b_h/2, dname, fontsize=8.2, fontweight='bold', color=c_title, va='center', zorder=4)

        # Dotted convergence line toward Feature Matrix
        fx_start = b_x + b_w
        fy_start = by + b_h/2
        fx_end = sec1_x + 2.38
        fy_end = y_sec + h_sec*0.50 + (by + b_h/2 - (y_sec + h_sec*0.50)) * 0.35
        ax.plot([fx_start, fx_start + 0.12, fx_end], [fy_start, fy_start, fy_end],
                color='#cbd5e1', lw=0.8, ls=':', zorder=2)

    # Feature Matrix X_t in R^(5054 x 62)
    mx = sec1_x + 2.40
    my = y_sec + 0.70
    mw = 1.55
    mh = 3.35

    ax.add_patch(Rectangle((mx, my), mw, mh, facecolor='#ffffff', edgecolor='#334155', lw=1.2, zorder=3))

    # Domain color bands (18 env, 8 host, 9 wild, 8 roads, 19 lags)
    c_widths = [mw * (18/62), mw * (8/62), mw * (9/62), mw * (8/62), mw * (19/62)]
    c_colors = ['#d1fae5', '#ffedd5', '#ede9fe', '#f1f5f9', '#fee2e2']
    c_borders = ['#10b981', '#fb923c', '#818cf8', '#94a3b8', '#f87171']
    
    cur_cx = mx
    for cw, cc, cb in zip(c_widths, c_colors, c_borders):
        ax.add_patch(Rectangle((cur_cx, my), cw, mh, facecolor=cc, edgecolor=cb, lw=0.5, alpha=0.6, zorder=3))
        cur_cx += cw

    # Horizontal row lines
    for ry in np.linspace(my, my + mh, 14)[1:-1]:
        ax.plot([mx, mx + mw], [ry, ry], color='#ffffff', lw=0.6, zorder=4)

    # Vertical column lines
    cur_cx = mx
    for cw in c_widths:
        cur_cx += cw
        ax.plot([cur_cx, cur_cx], [my, my + mh], color='#475569', lw=0.8, zorder=4)

    # Row indicators on left
    row_labels = [
        (my + mh - 0.22, r"$\text{CSD}_1$"),
        (my + mh - 0.44, r"$\text{CSD}_2$"),
        (my + mh - 0.66, r"$\text{CSD}_3$"),
        (my + mh*0.52, r"$\vdots$"),
        (my + 0.48, r"$\text{CSD}_i$"),
        (my + 0.24, r"$\vdots$"),
        (my + 0.05, r"$\text{CSD}_{5054}$")
    ]
    for ry, rtxt in row_labels:
        ax.text(mx - 0.06, ry, rtxt, ha='right', va='center', fontsize=6.2, color='#475569', zorder=5)

    # Highlighted target row i
    ax.add_patch(Rectangle((mx, my + 0.38), mw, 0.20, facecolor='#38bdf8', edgecolor='#0284c7', lw=1.0, alpha=0.7, zorder=5))
    ax.text(mx + mw/2, my + 0.48, r"$\mathbf{x}_{i,t} \in \mathbb{R}^{62}$", ha='center', va='center', fontsize=6.8, fontweight='bold', color='#082f49', zorder=6)

    # Matrix symbol
    ax.text(mx + mw/2, my + mh*0.72, r"$\mathbf{X}_t$", ha='center', va='center', fontsize=12.0, fontweight='bold', color='#1e293b', zorder=5)

    # Matrix dimension headers on top
    ax.text(mx + mw/2, my + mh + 0.16, "62 Features", ha='center', fontsize=7.8, fontweight='bold', color=c_title, zorder=5)
    ax.plot([mx, mx + mw], [my + mh + 0.08, my + mh + 0.08], color='#334155', lw=0.8, zorder=5)
    ax.plot([mx, mx], [my + mh + 0.05, my + mh + 0.11], color='#334155', lw=0.8, zorder=5)
    ax.plot([mx + mw, mx + mw], [my + mh + 0.05, my + mh + 0.11], color='#334155', lw=0.8, zorder=5)

    # Matrix Subtitle below
    ax.text(mx + mw/2, my - 0.18, "Monthly CSD Feature Matrix", ha='center', fontsize=7.6, fontweight='bold', color=c_title, zorder=5)
    ax.text(mx + mw/2, my - 0.36, r"$\mathbf{X}_t \in \mathbb{R}^{5054 \times 62}$", ha='center', fontsize=7.8, color='#0284c7', fontweight='bold', zorder=5)

    # Direct connection arrow from Feature Matrix to Spatial Graph
    flow_y_mat = y_sec + h_sec*0.50
    ax.text((mx + mw + sec2_x + 0.10)/2, flow_y_mat + 0.16, r"$\mathbf{x}_{v,t} \in \mathbf{X}_t$", ha='center', va='bottom', fontsize=7.2, fontweight='bold', color='#0284c7', zorder=8)

    # =========================================================================
    # SECTION 2: SPATIAL GRAPH LEARNING (MAIN FOCUS)
    # =========================================================================
    # 2A: Spatial Graph Construction (Left)
    # 2B: GraphSAGE Architecture with explicit Layer Operation Inset (Center)
    # 2C: Spatial Node Embedding Output (Right - fully enclosed in container)

    # -------------------------------------------------------------------------
    # 2A: SPATIAL GRAPH G = (V, E)
    # -------------------------------------------------------------------------
    gx_center = sec2_x + 1.25
    gy_center = y_sec + 2.65

    g_nodes = {
        'i': (gx_center, gy_center),                     # Target CSD i
        'j1': (gx_center + 0.65, gy_center + 0.62),      # Neighbor 1
        'j2': (gx_center - 0.60, gy_center + 0.58),      # Neighbor 2
        'j3': (gx_center + 0.10, gy_center - 0.85),      # Neighbor 3
        'u4': (gx_center - 0.95, gy_center - 0.40),
        'u5': (gx_center - 1.15, gy_center + 0.45),
        'u6': (gx_center - 0.45, gy_center + 1.25),
        'u7': (gx_center + 0.40, gy_center + 1.35),
        'u8': (gx_center + 1.10, gy_center + 0.55),
        'u9': (gx_center + 1.05, gy_center - 0.45),
        'u10': (gx_center - 0.65, gy_center - 1.15),
        'u11': (gx_center + 0.85, gy_center - 1.20),
        'u12': (gx_center - 1.25, gy_center - 1.00),
        'u13': (gx_center + 1.25, gy_center + 1.20)
    }

    # k=3 edges connecting target i to j1, j2, j3 (PROMINENT THICK BLUE)
    for neighbor in ['j1', 'j2', 'j3']:
        p1 = g_nodes['i']
        p2 = g_nodes[neighbor]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='#0284c7', lw=2.4, zorder=3)

    bg_edges = [
        ('j2', 'u5'), ('u5', 'u4'), ('u4', 'u10'), ('u10', 'j3'), ('j3', 'u11'),
        ('u11', 'u9'), ('u9', 'j1'), ('j1', 'u8'), ('u8', 'u13'), ('u13', 'u7'),
        ('u7', 'u6'), ('u6', 'j2'), ('u5', 'u6'), ('u4', 'u12'), ('u10', 'u12'),
        ('j1', 'u7'), ('j2', 'u4'), ('j3', 'u9'), ('u8', 'u7')
    ]
    for n1, n2 in bg_edges:
        p1 = g_nodes[n1]
        p2 = g_nodes[n2]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='#cbd5e1', lw=0.9, ls='-', zorder=2)

    for k in ['u4', 'u5', 'u6', 'u7', 'u8', 'u9', 'u10', 'u11', 'u12', 'u13']:
        pos = g_nodes[k]
        ax.add_patch(Circle(pos, 0.08, facecolor='#f1f5f9', edgecolor='#94a3b8', lw=0.8, zorder=4))

    for k, lbl, pos_adj in [('j1', r"$j_1$", (0.0, 0.0)), ('j2', r"$j_2$", (0.0, 0.0)), ('j3', r"$j_3$", (0.0, 0.0))]:
        pos = g_nodes[k]
        ax.add_patch(Circle(pos, 0.15, facecolor='#e0f2fe', edgecolor='#0284c7', lw=1.3, zorder=5))
        ax.text(pos[0] + pos_adj[0], pos[1] + pos_adj[1], lbl, ha='center', va='center', fontsize=7.4, fontweight='bold', color='#0369a1', zorder=6)

    # Target Node i
    pos_i = g_nodes['i']
    ax.add_patch(Circle(pos_i, 0.19, facecolor='#0284c7', edgecolor='#ffffff', lw=1.6, zorder=6))
    ax.text(pos_i[0], pos_i[1], r"$i$", ha='center', va='center', fontsize=9.2, fontweight='bold', color='white', zorder=7)

    # Node Feature labels on graph nodes
    ax.text(pos_i[0] - 0.28, pos_i[1] - 0.16, r"$\mathbf{x}_{i,t}$", fontsize=7.4, fontweight='bold', color='#0284c7', zorder=6)
    ax.text(g_nodes['j1'][0] + 0.22, g_nodes['j1'][1] + 0.10, r"$\mathbf{x}_{j_1,t}$", fontsize=6.8, color='#0369a1', zorder=6)
    ax.text(g_nodes['j2'][0] - 0.42, g_nodes['j2'][1] + 0.10, r"$\mathbf{x}_{j_2,t}$", fontsize=6.8, color='#0369a1', zorder=6)
    ax.text(g_nodes['j3'][0] + 0.22, g_nodes['j3'][1] - 0.10, r"$\mathbf{x}_{j_3,t}$", fontsize=6.8, color='#0369a1', zorder=6)

    # Simplified Graph Specification Card below Graph (as requested: only G, kNN, Centroid proximity)
    gc_w = 2.10
    gc_h = 0.72
    gc_x = gx_center - gc_w/2
    gc_y = y_sec + 0.28
    ax.add_patch(Rectangle((gc_x, gc_y), gc_w, gc_h, facecolor='#f8fafc', edgecolor='#cbd5e1', lw=0.8, zorder=2))
    ax.text(gx_center, gc_y + gc_h - 0.18, r"Spatial Graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$", ha='center', fontsize=8.0, fontweight='bold', color=c_title, zorder=4)
    ax.text(gx_center, gc_y + gc_h - 0.40, r"$k\text{NN}, \; k = 3$", ha='center', fontsize=7.6, color='#0284c7', fontweight='bold', zorder=4)
    ax.text(gx_center, gc_y + gc_h - 0.58, "Centroid proximity", ha='center', fontsize=7.0, color=c_muted, zorder=4)

    # Arrow from Graph into GraphSAGE Architecture container
    ax.annotate("", xy=(sec2_x + 2.80, gy_center), xytext=(gx_center + 1.25, gy_center),
                arrowprops=dict(arrowstyle="-|>", color="#0284c7", lw=1.8, mutation_scale=12), zorder=6)

    # -------------------------------------------------------------------------
    # 2B: GRAPHSAGE COMPUTATIONAL ARCHITECTURE (Container enclosing layers + embedding)
    # -------------------------------------------------------------------------
    # Extended downward and to the right to enclose the entire embedding vector and labels
    gs_box_x = sec2_x + 2.85
    gs_box_w = 5.45
    gs_box_h = 4.54
    gs_box_y = y_sec + 0.08

    # GraphSAGE Main Container
    ax.add_patch(Rectangle((gs_box_x, gs_box_y), gs_box_w, gs_box_h, facecolor='#fbfcfe', edgecolor='#93c5fd', lw=0.9, zorder=2))
    ax.text(gs_box_x + 0.15, gs_box_y + gs_box_h - 0.22, "GraphSAGE Architecture", fontsize=8.4, fontweight='bold', color='#1e3a8a', va='center', zorder=4)
    ax.plot([gs_box_x + 0.15, gs_box_x + gs_box_w - 0.15], [gs_box_y + gs_box_h - 0.36, gs_box_y + gs_box_h - 0.36], color='#dbeafe', lw=0.8, zorder=3)

    # TOP INSET: Layer-Wise Operation (Clarifies neighborhood aggregation is part of EACH layer)
    ins_x = gs_box_x + 0.15
    ins_y = gs_box_y + gs_box_h - 1.82
    ins_w = gs_box_w - 0.30
    ins_h = 1.38

    ax.add_patch(Rectangle((ins_x, ins_y), ins_w, ins_h, facecolor='#ffffff', edgecolor='#cbd5e1', lw=0.8, zorder=3))
    ax.text(ins_x + 0.12, ins_y + ins_h - 0.16, "Inset: Layer-Wise Message Passing", fontsize=7.2, fontweight='bold', color='#0284c7', va='center', zorder=4)

    # Inset Step 1: Neighbor Features
    s1_w, s1_h = 0.82, 0.48
    s1_x = ins_x + 0.12
    s1_y = ins_y + 0.24
    ax.add_patch(Rectangle((s1_x, s1_y), s1_w, s1_h, facecolor='#f8fafc', edgecolor='#94a3b8', lw=0.7, zorder=4))
    ax.text(s1_x + s1_w/2, s1_y + s1_h*0.65, "Neighbor", ha='center', va='center', fontsize=6.4, fontweight='bold', color=c_title, zorder=5)
    ax.text(s1_x + s1_w/2, s1_y + s1_h*0.25, r"$\{\mathbf{h}_j^{(l-1)}\}$", ha='center', va='center', fontsize=6.2, color='#0284c7', zorder=5)

    # Arrow to Mean Aggregation
    gap = 0.16
    ax.annotate("", xy=(s1_x + s1_w + gap - 0.02, s1_y + s1_h/2), xytext=(s1_x + s1_w + 0.02, s1_y + s1_h/2),
                arrowprops=dict(arrowstyle="-|>", color="#0284c7", lw=1.0, mutation_scale=8), zorder=5)

    # Inset Step 2: Mean Aggregation
    s2_w, s2_h = 1.25, 0.48
    s2_x = s1_x + s1_w + gap
    s2_y = s1_y
    ax.add_patch(Rectangle((s2_x, s2_y), s2_w, s2_h, facecolor='#eff6ff', edgecolor='#2563eb', lw=0.7, zorder=4))
    ax.text(s2_x + s2_w/2, s2_y + s2_h*0.65, r"Mean Aggregation: $\bigoplus$", ha='center', va='center', fontsize=6.3, fontweight='bold', color='#1d4ed8', zorder=5)
    ax.text(s2_x + s2_w/2, s2_y + s2_h*0.25, r"$\mathbf{h}_{\mathcal{N}(i)}^{(l)} = \frac{1}{k}\sum \mathbf{h}_j$", ha='center', va='center', fontsize=6.0, color=c_body, zorder=5)

    # Arrow to Concatenate
    ax.annotate("", xy=(s2_x + s2_w + gap - 0.02, s2_y + s2_h/2), xytext=(s2_x + s2_w + 0.02, s2_y + s2_h/2),
                arrowprops=dict(arrowstyle="-|>", color="#1d4ed8", lw=1.0, mutation_scale=8), zorder=5)

    # Inset Step 3: Concatenate with self feature
    s3_w, s3_h = 1.10, 0.48
    s3_x = s2_x + s2_w + gap
    s3_y = s1_y
    ax.add_patch(Rectangle((s3_x, s3_y), s3_w, s3_h, facecolor='#ffffff', edgecolor='#4338ca', lw=0.7, zorder=4))
    ax.text(s3_x + s3_w/2, s3_y + s3_h*0.65, "Concatenate", ha='center', va='center', fontsize=6.4, fontweight='bold', color='#3730a3', zorder=5)
    ax.text(s3_x + s3_w/2, s3_y + s3_h*0.25, r"$[\mathbf{h}_i^{(l-1)} \,\|\, \mathbf{h}_{\mathcal{N}(i)}^{(l)}]$", ha='center', va='center', fontsize=5.8, color='#3730a3', zorder=5)

    # Arrow to Linear + Activation
    ax.annotate("", xy=(s3_x + s3_w + gap - 0.02, s3_y + s3_h/2), xytext=(s3_x + s3_w + 0.02, s3_y + s3_h/2),
                arrowprops=dict(arrowstyle="-|>", color="#3730a3", lw=1.0, mutation_scale=8), zorder=5)

    # Inset Step 4: Linear + Activation
    s4_w, s4_h = 0.85, 0.48
    s4_x = s3_x + s3_w + gap
    s4_y = s1_y
    ax.add_patch(Rectangle((s4_x, s4_y), s4_w, s4_h, facecolor='#f0fdf4', edgecolor='#16a34a', lw=0.7, zorder=4))
    ax.text(s4_x + s4_w/2, s4_y + s4_h*0.65, r"Linear $+ \sigma$", ha='center', va='center', fontsize=6.4, fontweight='bold', color='#15803d', zorder=5)
    ax.text(s4_x + s4_w/2, s4_y + s4_h*0.25, r"$\mathbf{h}_i^{(l)}$", ha='center', va='center', fontsize=6.2, color='#15803d', zorder=5)

    # Compact GraphSAGE Equation below Inset (centered over macro layer blocks)
    lay_w = 3.65
    lay_h = 0.50
    lay_x = gs_box_x + 0.15

    eq_sage = r"$\mathbf{h}_i^{(l)} = \sigma\left(\mathbf{W}^{(l)} \left[\mathbf{h}_i^{(l-1)} \,\middle\|\, \frac{1}{|\mathcal{N}(i)|}\sum_{j \in \mathcal{N}(i)} \mathbf{h}_j^{(l-1)}\right]\right)$"
    ax.text(lay_x + lay_w/2, ins_y - 0.24, eq_sage, ha='center', va='center', fontsize=7.2, color='#1e3a8a', fontweight='bold', zorder=5)

    # MAIN MACRO ARCHITECTURE: Layer 1 -> Layer 2
    # Layer 1 Block
    l1_y = ins_y - 0.90
    ax.add_patch(Rectangle((lay_x, l1_y), lay_w, lay_h, facecolor='#ffffff', edgecolor='#1e40af', lw=1.1, zorder=4))
    ax.text(lay_x + 0.15, l1_y + lay_h*0.62, "GraphSAGE Layer 1", fontsize=7.8, fontweight='bold', color='#1e3a8a', va='center', zorder=5)
    ax.text(lay_x + 0.15, l1_y + lay_h*0.25, "Message passing over 3 nearest CSD neighbors", fontsize=6.6, color=c_muted, va='center', zorder=5)
    ax.text(lay_x + lay_w - 0.15, l1_y + lay_h/2, r"$\mathbf{h}_i^{(1)} \in \mathbb{R}^{64}$", fontsize=7.6, fontweight='bold', color='#1e40af', ha='right', va='center', zorder=5)

    # Arrow from Layer 1 to Layer 2
    ax.annotate("", xy=(lay_x + lay_w/2, l1_y - 0.38), xytext=(lay_x + lay_w/2, l1_y),
                arrowprops=dict(arrowstyle="-|>", color="#1e40af", lw=1.3, mutation_scale=10), zorder=5)

    # Layer 2 Block
    l2_y = l1_y - 0.38 - lay_h
    ax.add_patch(Rectangle((lay_x, l2_y), lay_w, lay_h, facecolor='#ffffff', edgecolor='#1e40af', lw=1.1, zorder=4))
    ax.text(lay_x + 0.15, l2_y + lay_h*0.62, "GraphSAGE Layer 2", fontsize=7.8, fontweight='bold', color='#1e3a8a', va='center', zorder=5)
    ax.text(lay_x + 0.15, l2_y + lay_h*0.25, "Second-hop spatial feature context aggregation", fontsize=6.6, color=c_muted, va='center', zorder=5)
    ax.text(lay_x + lay_w - 0.15, l2_y + lay_h/2, r"$\mathbf{z}_{i,t} \in \mathbb{R}^{64}$", fontsize=7.6, fontweight='bold', color='#1e40af', ha='right', va='center', zorder=5)

    # -------------------------------------------------------------------------
    # 2C: SPATIAL NODE EMBEDDING REPRESENTATION (Fully inside GraphSAGE container)
    # -------------------------------------------------------------------------
    # Vertically centered relative to GraphSAGE Layer 2 output
    l2_center_y = l2_y + lay_h/2
    emb_w = 0.46
    emb_h = 1.16
    emb_x = lay_x + lay_w + 0.62
    emb_y = l2_center_y - emb_h/2

    # Horizontal Arrow from Layer 2 to Embedding Vector (perfectly horizontal and centered)
    ax.annotate("", xy=(emb_x - 0.08, l2_center_y), xytext=(lay_x + lay_w, l2_center_y),
                arrowprops=dict(arrowstyle="-|>", color="#059669", lw=1.6, mutation_scale=12), zorder=5)

    # Embedding Vector visual block
    ax.add_patch(Rectangle((emb_x, emb_y), emb_w, emb_h, facecolor='#ffffff', edgecolor='#059669', lw=1.2, zorder=3))
    e_colors = ['#10b981', '#34d399', '#6ee7b7', '#059669', '#047857', '#10b981', '#34d399', '#059669']
    eh_cell = emb_h / len(e_colors)
    for ei, ecol in enumerate(e_colors):
        ax.add_patch(Rectangle((emb_x + 0.04, emb_y + (len(e_colors)-1-ei)*eh_cell + 0.02), emb_w - 0.08, eh_cell - 0.04,
                               facecolor=ecol, edgecolor='none', zorder=4))
    
    # Embedding Vector Labels (z_{i,t} strictly above, "Spatial Node Embedding" strictly below)
    ax.text(emb_x + emb_w/2, emb_y + emb_h + 0.22, r"$\mathbf{z}_{i,t}$", ha='center', va='center', fontsize=10.0, fontweight='bold', color='#047857', zorder=5)
    ax.text(emb_x + emb_w/2, emb_y + emb_h + 0.08, r"$\in \mathbb{R}^{64}$", ha='center', va='center', fontsize=7.2, color='#475569', zorder=5)
    ax.text(emb_x + emb_w/2, emb_y - 0.12, "Spatial Node", ha='center', va='center', fontsize=7.2, fontweight='bold', color=c_title, zorder=5)
    ax.text(emb_x + emb_w/2, emb_y - 0.26, "Embedding", ha='center', va='center', fontsize=7.2, fontweight='bold', color=c_title, zorder=5)

    # =========================================================================
    # SECTION 3: RISK FORECASTING (Simplified, clean alignment)
    # =========================================================================
    rf_x = sec3_x + 0.20
    rf_w = 2.10
    
    # Prediction Head Modules
    lin_w = 2.05
    lin_h = 0.52
    lin_x = sec3_x + (sec3_w - lin_w)/2
    lin_y = y_sec + 3.31

    # Clean orthogonal routing arrow from z_{i,t} into Linear Layer
    z_mid_x = emb_x + emb_w
    gap_x = (sec2_x + sec2_w + sec3_x) / 2
    ax.plot([z_mid_x, gap_x, gap_x, lin_x - 0.10],
            [l2_center_y, l2_center_y, lin_y + lin_h/2, lin_y + lin_h/2],
            color='#b91c1c', lw=1.6, zorder=5)
    ax.annotate("", xy=(lin_x, lin_y + lin_h/2), xytext=(lin_x - 0.10, lin_y + lin_h/2),
                arrowprops=dict(arrowstyle="-|>", color="#b91c1c", lw=1.6, mutation_scale=11), zorder=5)

    # Linear Layer Block
    ax.add_patch(Rectangle((lin_x, lin_y), lin_w, lin_h, facecolor='#ffffff', edgecolor='#b91c1c', lw=1.1, zorder=3))
    ax.text(lin_x + lin_w/2, lin_y + lin_h*0.64, "Linear Layer", ha='center', va='center', fontsize=7.8, fontweight='bold', color='#991b1b', zorder=4)
    ax.text(lin_x + lin_w/2, lin_y + lin_h*0.26, r"$\text{logit}_i = \mathbf{w}^T \mathbf{z}_{i,t} + b$", ha='center', va='center', fontsize=6.8, color=c_body, zorder=4)

    # Arrow from Linear to Sigmoid
    sig_h = 0.52
    sig_y = lin_y - 0.48 - sig_h
    sig_w = lin_w
    sig_x = lin_x
    ax.annotate("", xy=(sig_x + sig_w/2, sig_y + sig_h), xytext=(lin_x + lin_w/2, lin_y),
                arrowprops=dict(arrowstyle="-|>", color="#b91c1c", lw=1.3, mutation_scale=10), zorder=5)

    # Sigmoid Layer Block
    ax.add_patch(Rectangle((sig_x, sig_y), sig_w, sig_h, facecolor='#ffffff', edgecolor='#b91c1c', lw=1.1, zorder=3))
    ax.text(sig_x + sig_w/2, sig_y + sig_h*0.64, "Sigmoid", ha='center', va='center', fontsize=7.8, fontweight='bold', color='#991b1b', zorder=4)
    ax.text(sig_x + sig_w/2, sig_y + sig_h*0.26, r"$\hat{p}_{i,t+1} = \sigma(\text{logit}_i)$", ha='center', va='center', fontsize=6.8, color=c_body, zorder=4)

    # Arrow from Sigmoid into Output Panel
    out_h = 1.05
    out_y = sig_y - 0.58 - out_h
    out_w = lin_w
    out_x = lin_x
    ax.annotate("", xy=(out_x + out_w/2, out_y + out_h), xytext=(sig_x + sig_w/2, sig_y),
                arrowprops=dict(arrowstyle="-|>", color="#b91c1c", lw=1.4, mutation_scale=11), zorder=5)
    
    # Value indicator p̂_{i,t+1} on arrow
    ax.text(out_x + out_w/2 + 0.35, (sig_y + out_y + out_h)/2, r"$\hat{p}_{i,t+1}$", fontsize=7.6, fontweight='bold', color='#b91c1c', va='center', zorder=6)

    # Formal Output Block (Simplified as requested: no example numbers table)
    ax.add_patch(Rectangle((out_x, out_y), out_w, out_h, facecolor='#fffbfa', edgecolor='#b91c1c', lw=1.1, zorder=3))
    ax.text(out_x + out_w/2, out_y + out_h - 0.24, "HPAI Outbreak Risk", ha='center', va='center', fontsize=8.0, fontweight='bold', color='#991b1b', zorder=4)
    ax.text(out_x + out_w/2, out_y + out_h - 0.52, "Month t + 1", ha='center', va='center', fontsize=7.4, fontweight='bold', color=c_title, zorder=4)
    ax.text(out_x + out_w/2, out_y + 0.24, r"$\hat{p}_{i,t+1} = P(Y_{i,t+1} = 1)$", ha='center', va='center', fontsize=7.2, color='#b91c1c', zorder=4)

    ax.text(16.05, 0.16, r"Features at month $t$ $\longrightarrow$ Risk at month $t+1$",
            ha='right', va='center', fontsize=7.8, color='#0284c7', fontstyle='italic', fontweight='bold', zorder=5)

    # Save outputs across all manuscript directories
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dirs = [
        os.path.join(base_dir, "results", "manuscript_elsevier", "figures"),
        os.path.join(base_dir, "results", "manuscript_elsevier"),
        os.path.join(base_dir, "results", "manuscript_latex", "figures"),
        os.path.join(base_dir, "results", "manuscript_latex"),
        os.path.join(base_dir, "results", "final_manuscript", "figures"),
        os.path.join(base_dir, "results", "final_manuscript"),
        os.path.join(base_dir, "results", "manuscript_draft", "figures", "final"),
        os.path.join(base_dir, "scratch")
    ]

    import time
    def safe_save(fig_obj, path, **kwargs):
        for attempt in range(5):
            try:
                fig_obj.savefig(path, **kwargs)
                return
            except OSError as err:
                if attempt == 4:
                    print(f"Warning: could not save to {path}: {err}")
                else:
                    time.sleep(0.6)

    for d in out_dirs:
        os.makedirs(d, exist_ok=True)
        safe_save(fig, os.path.join(d, "Figure1_schematic.png"), dpi=300)
        safe_save(fig, os.path.join(d, "Figure1_schematic.pdf"))
        safe_save(fig, os.path.join(d, "Figure1_study_design.png"), dpi=300)
        safe_save(fig, os.path.join(d, "Figure1_study_design.pdf"))

    print("Simplified scientific architecture diagram generated successfully!")
    plt.close()

if __name__ == '__main__':
    generate_figure()
