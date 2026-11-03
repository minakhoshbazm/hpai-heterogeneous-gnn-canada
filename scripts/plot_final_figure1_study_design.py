import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "results/manuscript_draft/figures/final"
FIG_DIR.mkdir(parents=True, exist_ok=True)

def draw_box(ax, xy, width, height, text, facecolor='#e6f2ff', edgecolor='#1f77b4'):
    box = mpatches.FancyBboxPatch(
        xy, width, height, boxstyle="round,pad=0.05",
        facecolor=facecolor, edgecolor=edgecolor, linewidth=2, zorder=2
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2, xy[1] + height / 2, text,
        ha='center', va='center', fontsize=10, family='sans-serif', zorder=3
    )

def plot_figure_1():
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis('off')

    # Data sources (Left column)
    sources = [
        "Remote sensing /\nland cover",
        "Climate /\nERA5-Land",
        "Poultry and\nhost density",
        "Roads /\ninfrastructure",
        "Poultry outbreak\nhistory",
        "Wild-bird HPAI\nsurveillance"
    ]
    
    source_width = 2.0
    source_height = 0.6
    x_left = 0.5
    y_start = 5.5
    y_step = -0.8
    
    source_coords = []
    for i, src in enumerate(sources):
        y = y_start + i * y_step
        draw_box(ax, (x_left, y), source_width, source_height, src)
        source_coords.append((x_left + source_width, y + source_height / 2))

    # Middle column (Integration and Graph)
    mid_width = 3.0
    mid_height = 1.0
    x_mid = 3.5
    
    y_feat = 4.2
    feat_text = "Monthly CSD feature integration\n(62 features)\n(5,054 CSDs)"
    draw_box(ax, (x_mid, y_feat), mid_width, mid_height, feat_text, facecolor='#fff2e6', edgecolor='#ff7f0e')
    
    y_graph = 2.4
    graph_text = "kNN-3 geographic graph\n(9,655 unique undirected edges)"
    draw_box(ax, (x_mid, y_graph), mid_width, mid_height, graph_text, facecolor='#fff2e6', edgecolor='#ff7f0e')

    # Lines from sources to features
    for cy in source_coords:
        ax.plot([cy[0], x_mid], [cy[1], y_feat + mid_height/2], color='gray', linestyle='-', linewidth=1.5, zorder=1)

    # Line from features to graph
    ax.annotate("", xy=(x_mid + mid_width/2, y_graph + mid_height), 
                xytext=(x_mid + mid_width/2, y_feat),
                arrowprops=dict(arrowstyle="->", color='gray', lw=2))

    # Right column (Model and Output)
    right_width = 2.5
    right_height = 1.0
    x_right = 7.2
    
    y_model = 3.3
    model_text = "2-layer GraphSAGE\n(Spatial message passing)"
    draw_box(ax, (x_right, y_model), right_width, right_height, model_text, facecolor='#e6ffe6', edgecolor='#2ca02c')

    # Line from graph to model
    ax.annotate("", xy=(x_right, y_model + right_height/2), 
                xytext=(x_mid + mid_width, y_graph + mid_height/2),
                arrowprops=dict(arrowstyle="->", color='gray', lw=2))
                
    # Line from features to model (since it uses features too)
    ax.annotate("", xy=(x_right, y_model + right_height/2 + 0.2), 
                xytext=(x_mid + mid_width, y_feat + mid_height/2),
                arrowprops=dict(arrowstyle="->", color='gray', lw=2))

    y_out = 1.2
    out_text = "Risk score for outbreak\nin month t+1"
    draw_box(ax, (x_right, y_out), right_width, right_height, out_text, facecolor='#f9f2fc', edgecolor='#9467bd')

    # Line from model to output
    ax.annotate("", xy=(x_right + right_width/2, y_out + right_height), 
                xytext=(x_right + right_width/2, y_model),
                arrowprops=dict(arrowstyle="->", color='gray', lw=2))

    # Bottom Timeline
    timeline_y = 0.2
    ax.plot([0.5, 9.5], [timeline_y+0.4, timeline_y+0.4], color='black', linewidth=3)
    
    # Train
    ax.text(2.0, timeline_y, "2022–2023 TRAIN\n(262 positive CSD-months)", ha='center', fontsize=10)
    # Val
    ax.text(5.0, timeline_y, "2024 VALIDATION\n(40 positive CSD-months)", ha='center', fontsize=10)
    # Test
    ax.text(8.0, timeline_y, "2025 LOCKED TEST\n(79 positive CSD-months)", ha='center', fontsize=10, weight='bold')

    for ext in ['png', 'pdf', 'svg']:
        plt.savefig(FIG_DIR / f"Figure1_study_design.{ext}", bbox_inches='tight', dpi=300)
    plt.close()
    print("Figure 1 generated.")

if __name__ == "__main__":
    plot_figure_1()
