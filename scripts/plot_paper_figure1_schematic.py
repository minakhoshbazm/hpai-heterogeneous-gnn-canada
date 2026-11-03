import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def generate_figure1():
    # Set high DPI and elegant fonts
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
    
    fig, ax = plt.subplots(figsize=(16, 9), dpi=300)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Background canvas tint
    bg = patches.Rectangle((0, 0), 16, 10, facecolor='#fafbfc', zorder=0)
    ax.add_patch(bg)

    # Title Banner
    banner = patches.FancyBboxPatch((0.5, 9.0), 15.0, 0.8, boxstyle="round,pad=0.08",
                                   facecolor='#1b3b5c', edgecolor='none', zorder=1)
    ax.add_patch(banner)
    ax.text(8.0, 9.4, "Nationwide Spatial Graph Deep Learning for 1-Month-Ahead Poultry HPAI Outbreak Prediction",
            ha='center', va='center', color='white', fontsize=15, fontweight='bold', zorder=2)

    # Helper function for stage cards
    def create_card(x, y, w, h, title, color_header, color_bg, border_color):
        card = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                                      facecolor=color_bg, edgecolor=border_color, linewidth=2.0, zorder=1)
        ax.add_patch(card)
        # Header banner
        header = patches.FancyBboxPatch((x, y + h - 0.7), w, 0.7, boxstyle="round,pad=0.05",
                                        facecolor=color_header, edgecolor='none', zorder=2)
        ax.add_patch(header)
        ax.text(x + w/2, y + h - 0.35, title, ha='center', va='center',
                color='white', fontsize=12, fontweight='bold', zorder=3)

    # Helper function for items
    def add_subbox(x, y, w, h, title, desc, icon_color='#e2e8f0', text_color='#1e293b', title_fs=10.0, desc_fs=8.5):
        box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04",
                                     facecolor='white', edgecolor='#cbd5e1', linewidth=1.2, zorder=2)
        ax.add_patch(box)
        ax.text(x + 0.15, y + h - 0.18, title, ha='left', va='top',
                color=text_color, fontsize=title_fs, fontweight='bold', zorder=3)
        ax.text(x + 0.15, y + h - 0.45, desc, ha='left', va='top',
                color='#334155', fontsize=desc_fs, linespacing=1.22, zorder=3)

    # --- Column 1: Multi-Modal Geospatial Features (x=0.6, w=3.4) ---
    create_card(0.6, 0.6, 3.5, 8.0, "1. Multi-Modal Ingestion", "#2563eb", "#eff6ff", "#93c5fd")
    ax.text(2.35, 7.5, "62 Spatiotemporal Covariates", ha='center', fontsize=10.5, fontweight='bold', color='#1e40af')
    
    add_subbox(0.8, 6.4, 3.1, 0.8, "Land Cover (Dynamic World)", "Near real-time 10m satellite rasters")
    add_subbox(0.8, 5.4, 3.1, 0.8, "Climate (ERA5-Land)", "Temperature, soil moisture, humidity")
    add_subbox(0.8, 4.4, 3.1, 0.8, "Poultry Density (GLW4 & Ag Census)", "Commercial broilers, turkeys, layers")
    add_subbox(0.8, 3.4, 3.1, 0.8, "Road Infrastructure (NRN)", "Highway distance, primary route density")
    add_subbox(0.8, 2.4, 3.1, 0.8, "Wild-Bird Biosurveillance", "eBird waterfowl & positive detections")
    add_subbox(0.8, 1.4, 3.1, 0.8, "Historical Outbreak Lags", "1, 2, 3, 6, 12-month local persistence")
    add_subbox(0.8, 0.8, 3.1, 0.5, "5,054 CSDs x 47 Calendar Months", "Total: 237,538 CSD-months nationwide", icon_color='#dbeafe', text_color='#1d4ed8')

    # --- Column 2: Forward-Time Temporal Splitting (x=4.4, w=3.2) ---
    create_card(4.4, 0.6, 3.3, 8.0, "2. Forward Splitting", "#0891b2", "#ecfeff", "#a5f3fc")
    ax.text(6.05, 7.5, "Strict Anti-Leakage Protocol", ha='center', fontsize=10.5, fontweight='bold', color='#0e7490')

    add_subbox(4.6, 5.7, 2.9, 1.4, "Training Period (2022-2023)",
               "• 24 monthly snapshots\n• 121,296 CSD-months\n• 262 positive outbreaks\n• Prevalence: 0.216%",
               text_color='#0f766e')
    
    add_subbox(4.6, 3.8, 2.9, 1.4, "Validation Period (2024)",
               "• 12 monthly snapshots\n• 60,648 CSD-months\n• 40 positive outbreaks\n• Hyperparameter tuning",
               text_color='#0284c7')

    add_subbox(4.6, 1.9, 2.9, 1.4, "Locked Test Period (2025)",
               "• 11 monthly snapshots\n• 55,594 CSD-months\n• 79 positive outbreaks\n• Zero development access",
               text_color='#b91c1c')

    add_subbox(4.6, 0.8, 2.9, 0.8, "Decision Window", "Month t features predict month t+1", icon_color='#cffafe', text_color='#0369a1')

    # --- Column 3: Graph Construction & GNN (x=8.0, w=3.5) ---
    create_card(8.0, 0.6, 3.6, 8.0, "3. Spatial GraphSAGE", "#7c3aed", "#faf5ff", "#ddd6fe")
    ax.text(9.8, 7.5, "Inductive Neighborhood Learning", ha='center', fontsize=10.5, fontweight='bold', color='#6d28d9')

    add_subbox(8.2, 6.2, 3.2, 1.0, "Centroid kNN-3 Topology", "5,054 nodes | 9,655 undirected edges\nConnects immediate agricultural neighbors", text_color='#5b21b6')
    add_subbox(8.2, 4.9, 3.2, 1.1, "Layer 1: 1-Hop Message Passing", "h_v^(1) = σ(W1 · [x_v || Mean_{u∈N(v)} x_u])\nAggregates local landscape & wild bird pressure", text_color='#4c1d95')
    add_subbox(8.2, 3.6, 3.2, 1.1, "Layer 2: 2-Hop Receptive Field", "h_v^(2) = σ(W2 · [h_v^(1) || Mean_{u∈N(v)} h_u^(1)])\nCaptures broad regional corridor context", text_color='#4c1d95')
    add_subbox(8.2, 2.3, 3.2, 1.1, "Flyway-Dynamic Formulation", "W_ij(t) modulated by migratory phenology\n+33% spring & +30% autumn corridor weight", text_color='#6d28d9')
    add_subbox(8.2, 0.8, 3.2, 1.3, "Architectural Parsimony", "Added recurrent, transfer, or\nSMOTE schemes yield no reliable\nimprovement under label scarcity", text_color='#4338ca')

    # --- Column 4: Operational Evaluation & Decision Support (x=11.9, w=3.5) ---
    create_card(11.9, 0.6, 3.5, 8.0, "4. Biosurveillance Utility", "#059669", "#f0fdf4", "#bbf7d0")
    ax.text(13.65, 7.5, "Veterinary Operational Impact", ha='center', fontsize=10.5, fontweight='bold', color='#047857')

    add_subbox(12.1, 5.9, 3.1, 1.35, "Controlled 1:5 Benchmark",
               "• 79 Outbreaks : 395 Background (N=474)\n• GraphSAGE PR-AUC: 0.7374 ± 0.024\n• RF: 0.7248 | Matched MLP: 0.7063\n• Precision@10: 0.954 ± 0.038",
               text_color='#065f46')

    add_subbox(12.1, 4.3, 3.1, 1.45, "Spatial Lift Ablation (ΔPR-AUC)",
               "• 1:5 Full: Δ = +0.0328 | 1:5 No-lag: Δ = +0.0537\n• Commercial Full: Δ = +0.0253 | No-lag: Δ = +0.0227\n• National Full: Δ = +0.0403 | No-lag: Δ = +0.0298",
               text_color='#047857')

    add_subbox(12.1, 2.3, 3.1, 1.85, "Prospective Operational Result",
               "• Predefined Commercial Domain (2,222 CSDs/mo)\n• K_monthly = 25 CSDs/mo (275 annual audits)\n• GraphSAGE: 16 captures (35.6% recall)\n• Random Forest: 12 captures (+4 lift / +33%)\n• Illustrative utility lift: +$4.00M CAD over RF",
               text_color='#065f46')

    add_subbox(12.1, 0.8, 3.1, 1.35, "Explainable AI (XAI)",
               "• Saliency: Host (31.8%), Lags (25.4%), Water (18.2%)\n• 34.2% normalized spatial attribution mass\n  transmitted via 1-hop geographic edges",
               text_color='#064e3b')

    # Arrows between columns
    arrow_style = dict(arrowstyle="->,head_length=0.4,head_width=0.25", color="#475569", lw=2.5)
    ax.annotate("", xy=(4.35, 4.6), xytext=(4.15, 4.6), arrowprops=arrow_style)
    ax.annotate("", xy=(7.95, 4.6), xytext=(7.75, 4.6), arrowprops=arrow_style)
    ax.annotate("", xy=(11.85, 4.6), xytext=(11.65, 4.6), arrowprops=arrow_style)

    plt.tight_layout()
    
    out_dirs = [
        "results/manuscript_elsevier/figures",
        "results/manuscript_elsevier",
        "results/manuscript_latex/figures",
        "results/manuscript_latex",
        "results/final_manuscript/figures"
    ]
    for d in out_dirs:
        os.makedirs(d, exist_ok=True)
        fig.savefig(os.path.join(d, "Figure1_schematic.png"), dpi=300, bbox_inches='tight')
        fig.savefig(os.path.join(d, "Figure1_schematic.pdf"), bbox_inches='tight')
        fig.savefig(os.path.join(d, "Figure1_study_design.png"), dpi=300, bbox_inches='tight')
        fig.savefig(os.path.join(d, "Figure1_study_design.pdf"), bbox_inches='tight')
    plt.close()
    print("Figure 1 generated successfully!")

if __name__ == '__main__':
    generate_figure1()
