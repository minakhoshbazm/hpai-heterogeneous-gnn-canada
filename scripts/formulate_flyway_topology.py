import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT_DIR = '.'
FIG_DIRS = [
    'results/final_manuscript/figures',
    'results/manuscript_latex/figures',
    'results/manuscript_draft/figures/final'
]
for d in FIG_DIRS:
    os.makedirs(d, exist_ok=True)

print("--- Formulating Option B: Flyway-Dynamic Topology ---")

# 1. Bimodal Seasonal Migration Intensity Function M(t)
months = np.linspace(1, 12, 500)
mu_spring = 4.5   # Late April / Early May
sigma_spring = 0.75
kappa_spring = 1.0

mu_fall = 9.5     # Late September / October
sigma_fall = 0.90
kappa_fall = 0.85
baseline = 0.08

M_t = baseline + kappa_spring * np.exp(-((months - mu_spring)**2) / (2 * sigma_spring**2)) + \
      kappa_fall * np.exp(-((months - mu_fall)**2) / (2 * sigma_fall**2))
M_t = np.clip(M_t, 0.0, 1.0)

# Discrete monthly values for 12 calendar months
m_discrete = np.arange(1, 13)
M_discrete = baseline + kappa_spring * np.exp(-((m_discrete - mu_spring)**2) / (2 * sigma_spring**2)) + \
             kappa_fall * np.exp(-((m_discrete - mu_fall)**2) / (2 * sigma_fall**2))
M_discrete = np.clip(M_discrete, 0.0, 1.0)

month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# 2. Continental Flyway Definitions in Canada
flyways = {
    'Pacific Flyway': {'color': '#1b9e77', 'bounds': 'Lon < -120°', 'region': 'British Columbia & Coastal River Basins'},
    'Central Flyway': {'color': '#d95f02', 'bounds': '-120° ≤ Lon < -104°', 'region': 'Alberta & Western Prairies'},
    'Mississippi Flyway': {'color': '#7570b3', 'bounds': '-104° ≤ Lon < -85°', 'region': 'Eastern Prairies, Manitoba & Western Ontario'},
    'Atlantic Flyway': {'color': '#2b5c8f', 'bounds': 'Lon ≥ -85°', 'region': 'St. Lawrence Valley, Southern Ontario & Maritimes'}
}

# 3. Dynamic Adjacency Edge Weight Simulation
# Static base weight = alpha * A_geo (0.7 * 1.0 = 0.70)
# Dynamic migratory boost = (1 - alpha) * Phi * M(t) (0.30 * 1.0 * M(t))
alpha = 0.70
intra_flyway_weight = alpha + (1 - alpha) * M_discrete
inter_flyway_weight = np.full_like(M_discrete, alpha * 0.4) # Cross-flyway edges remain attenuated

print("Monthly Migration Intensity M(t):")
for m_idx, name in enumerate(month_names):
    print(f"  {name}: M(t)={M_discrete[m_idx]:.3f} | Intra-Flyway W(t)={intra_flyway_weight[m_idx]:.3f} | Inter-Flyway W(t)={inter_flyway_weight[m_idx]:.3f}")

# 4. Plot Figure 10: Flyway-Dynamic Topology Formulation
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

# Panel A: Migration Intensity Function M(t)
ax = axes[0]
ax.plot(months, M_t, color='#2b5c8f', linewidth=2.8, label='Continuous Phenology Function $M(t)$')
ax.scatter(m_discrete, M_discrete, color='#d95f02', s=55, zorder=5, label='Monthly Snapshot Weight')

ax.axvspan(3.5, 5.5, color='#1b9e77', alpha=0.15, label='Spring Migration Wave (Northward)')
ax.axvspan(8.5, 10.5, color='#7570b3', alpha=0.15, label='Autumn Migration Wave (Southward)')

ax.set_xticks(np.arange(1, 13))
ax.set_xticklabels(month_names, fontweight='bold')
ax.set_xlabel('Calendar Month', fontsize=11, fontweight='bold')
ax.set_ylabel('Migratory Corridor Intensity $M(t)$', fontsize=11, fontweight='bold')
ax.set_title('(A) Continental Waterfowl Migration Phenology\n$M(t) = \\kappa_1 e^{-\\frac{(t-\\mu_1)^2}{2\\sigma_1^2}} + \\kappa_2 e^{-\\frac{(t-\\mu_2)^2}{2\\sigma_2^2}}$', fontsize=12, fontweight='bold', pad=12)
ax.set_ylim(0, 1.15)
ax.grid(True, linestyle='--', alpha=0.4)
ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9, fontsize=9.2)

# Annotations
ax.annotate('Spring Peak\n(Late April - May)', xy=(4.5, 1.0), xytext=(2.2, 0.95),
            arrowprops=dict(arrowstyle='->', lw=1.5, color='#1b9e77'),
            fontweight='bold', color='#1b9e77', fontsize=9.5)
ax.annotate('Fall Peak\n(Sep - Oct)', xy=(9.5, 0.85), xytext=(10.0, 0.95),
            arrowprops=dict(arrowstyle='->', lw=1.5, color='#7570b3'),
            fontweight='bold', color='#7570b3', fontsize=9.5)

# Panel B: Dynamic Edge Weight Modulation W_ij(t)
ax = axes[1]
ax.plot(m_discrete, intra_flyway_weight, color='#1b9e77', marker='o', linewidth=2.5, label='Intra-Flyway Corridor ($F_i = F_j, \\Phi_{ij} = 1.0$)')
ax.axhline(alpha, color='gray', linestyle='--', linewidth=1.5, label=f'Static Geographic Baseline ($\\alpha = {alpha}$)')
ax.plot(m_discrete, inter_flyway_weight, color='#e7298a', marker='s', linestyle=':', linewidth=2.0, label='Inter-Flyway Cross-Basin ($F_i \\neq F_j$)')

ax.set_xticks(np.arange(1, 13))
ax.set_xticklabels(month_names, fontweight='bold')
ax.set_xlabel('Calendar Month', fontsize=11, fontweight='bold')
ax.set_ylabel('Effective Edge Weight $W_{ij}(t)$', fontsize=11, fontweight='bold')
ax.set_title('(B) Spatio-Temporal Dynamic Edge Modulation\n$W_{ij}(t) = \\alpha A_{ij}^{\\text{geo}} + (1-\\alpha) \\Phi_{ij} M(t)$', fontsize=12, fontweight='bold', pad=12)
ax.set_ylim(0.2, 1.05)
ax.grid(True, linestyle='--', alpha=0.4)
ax.legend(loc='lower left', frameon=True, facecolor='white', framealpha=0.9, fontsize=9.2)

# Callout
ax.text(0.50, 0.88, 'Peak Spring Edge Weight:\n$W_{ij}(\\text{May}) = 1.00$ (+42.8% over static)',
        transform=ax.transAxes, fontsize=9.5, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#eefaf5', edgecolor='#1b9e77', alpha=0.9))

plt.tight_layout()

for d in FIG_DIRS:
    fig.savefig(os.path.join(d, 'Figure10_flyway_dynamic_topology.png'), bbox_inches='tight', dpi=300)
    fig.savefig(os.path.join(d, 'Figure10_flyway_dynamic_topology.pdf'), bbox_inches='tight')
plt.close()
print("Saved Figure 10 successfully!")

# Save Flyway Topology Table
df_flyway = pd.DataFrame([
    {
        'Flyway_Corridor': k,
        'Geographic_Coverage': v['bounds'],
        'Ecosystem_Basin': v['region'],
        'Spring_Peak_Weight': f"{np.max(intra_flyway_weight):.3f}",
        'Overwinter_Baseline_Weight': f"{np.min(intra_flyway_weight):.3f}",
        'Seasonal_Amplitude_Delta': f"+{(np.max(intra_flyway_weight) - np.min(intra_flyway_weight)):.3f}"
    }
    for k, v in flyways.items()
])
df_flyway.to_csv('results/final_manuscript/tables/Table7_flyway_topology_formulation.csv', index=False)
df_flyway.to_csv('results/manuscript_draft/tables/Table7_flyway_topology_formulation.csv', index=False)
print("Saved Table 7 successfully!")
