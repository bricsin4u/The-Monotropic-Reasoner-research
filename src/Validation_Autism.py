
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# --- Configuration ---
N_AGENTS = 5000
RANDOM_SEED = 42
MELTDOWN_THRESHOLD_G = 2.0

REPO_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = REPO_ROOT / "figures"

# --- G-Model Functions (Adapted for Neurodivergence) ---

def calculate_g(I, A, D, S, B_err, B_mot_base, C, E, agent_type):
    """
    Calculates Stupidity (G) based on the formal model, with specific adaptations
    for Neurotypical (NT) vs Neurodivergent (ND) architectures.
    """
    
    # 1. Processing Error (Cognitive Noise)
    # ND agents often have higher 'local' precision (lower B_err) if attentive.
    term_1 = B_err / I 

    # 2. Motivated Bias (Ideology/Social Conformity)
    # Hypothesis: ND agents have LOWER social motivated bias.
    # B_mot is driven by Social Pressure (S) for NTs, but less so for NDs.
    if agent_type == 'ND':
        B_mot = B_mot_base * 0.5  # NDs less prone to social groupthink
    else:
        B_mot = B_mot_base * (1 + 0.5 * S) # NTs conform more

    # 3. Environmental Load (The Core Difference)
    
    # a) Effective Noise (D_eff)
    # NT: Linear filtering.
    # ND: High-Precision (Exponential sensitivity to noise).
    if agent_type == 'ND':
         # "The Sensory Cliff": Noise scales faster for ND
        D_eff = D * np.exp(max(0, D - 0.4)) # Threshold is lower (0.4 vs 0.7)
    else:
        D_eff = D * np.exp(max(0, D - 0.7)) # Standard Threshold
        
    # b) Masking Cost (The Hidden Variable)
    # For ND agents, High Social Pressure (S) consumes Attention (A).
    # A_available = A_raw - Masking_Cost
    if agent_type == 'ND':
        masking_tax = 0.4 * S # Significant tax
    else:
        masking_tax = 0.05 * S # Minimal tax (natural social instincts)
    
    A_eff = max(0.01, A - masking_tax)

    term_2 = D_eff / A_eff

    # 4. Social/Emotional Regulation
    term_3 = (S * (1 - C)) / E

    # Total G
    G = (0.3 * (term_1 + B_mot)) + (0.5 * term_2) + (0.2 * term_3)
    return G, A_eff, D_eff

# --- Simulation ---
np.random.seed(RANDOM_SEED)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

results = []

for _ in range(N_AGENTS):
    # Shared Environmental vars
    D_env = np.random.beta(2, 2) # Noise typically distributed around 0.5
    S_env = np.random.beta(2, 2) # Social pressure
    
    # --- Agent A: Neurotypical (Polytropic) ---
    I_nt = np.random.normal(100, 15)
    A_nt = np.random.beta(3, 3) # Moderate, balanced attention
    B_err_nt = np.random.beta(2, 5)
    B_mot_nt = np.random.beta(2, 2)
    
    G_nt, A_eff_nt, D_eff_nt = calculate_g(
        I=I_nt/100, A=A_nt, D=D_env, S=S_env, 
        B_err=B_err_nt, B_mot_base=B_mot_nt, 
        C=0.5, E=0.6, agent_type='NT'
    )
    
    results.append({
        'Type': 'Neurotypical',
        'G': G_nt,
        'Environment_Noise': D_env,
        'Social_Pressure': S_env,
        'Raw_Attention': A_nt,
        'Effective_Attention': A_eff_nt,
        'Effective_Load': D_eff_nt
    })

    # --- Agent B: Autistic (Monotropic) ---
    I_nd = np.random.normal(100, 15) # Example: Same intelligence distribution
    A_nd = np.random.beta(5, 2) # High raw attention (Monotropic focus)
    B_err_nd = np.random.beta(1, 5) # Lower processing error (Detail oriented)
    B_mot_nd = np.random.beta(1, 4) # Lower motivated bias (Truth seeking)
    
    G_nd, A_eff_nd, D_eff_nd = calculate_g(
        I=I_nd/100, A=A_nd, D=D_env, S=S_env, 
        B_err=B_err_nd, B_mot_base=B_mot_nd, 
        C=0.5, E=0.4, agent_type='ND' # Slightly lower E (emotional regulation difficulty)
    )
    
    results.append({
        'Type': 'Autistic',
        'G': G_nd,
        'Environment_Noise': D_env,
        'Social_Pressure': S_env,
        'Raw_Attention': A_nd,
        'Effective_Attention': A_eff_nd,
        'Effective_Load': D_eff_nd
    })

df = pd.DataFrame(results)

# --- Analysis & Stats ---

def mean_ci_95(values: pd.Series) -> tuple[float, float, float]:
    clean = values.dropna().to_numpy()
    n = clean.size
    mean = float(clean.mean())
    if n < 2:
        return mean, mean, mean
    se = float(clean.std(ddof=1) / np.sqrt(n))
    half_width = 1.96 * se
    return mean, mean - half_width, mean + half_width

# 1. Performance in Low Noise (The Savant Zone)
low_noise = df[df['Environment_Noise'] < 0.3]
g_low_nt, g_low_nt_lo, g_low_nt_hi = mean_ci_95(low_noise[low_noise['Type'] == 'Neurotypical']['G'])
g_low_nd, g_low_nd_lo, g_low_nd_hi = mean_ci_95(low_noise[low_noise['Type'] == 'Autistic']['G'])

# 2. Performance in High Noise (The Burnout Zone)
high_noise = df[df['Environment_Noise'] > 0.7]
g_high_nt = high_noise[high_noise['Type'] == 'Neurotypical']['G'].mean()
g_high_nd = high_noise[high_noise['Type'] == 'Autistic']['G'].mean()

# 3. Impact of Social Pressure on Attention
high_social = df[df['Social_Pressure'] > 0.8]
att_drop_nt = (high_social[high_social['Type'] == 'Neurotypical']['Raw_Attention'] - high_social[high_social['Type'] == 'Neurotypical']['Effective_Attention']).mean()
att_drop_nd = (high_social[high_social['Type'] == 'Autistic']['Raw_Attention'] - high_social[high_social['Type'] == 'Autistic']['Effective_Attention']).mean()


print("--- SIMULATION RESULTS ---")
print(f"Low Noise (D<0.3) G-Score (Lower is better):")
print(f"  NT: {g_low_nt:.3f}")
print(f"  ND: {g_low_nd:.3f} -> ND is {((g_low_nt - g_low_nd)/g_low_nt)*100:.1f}% MORE rational.")
print(f"  NT 95% CI: [{g_low_nt_lo:.3f}, {g_low_nt_hi:.3f}]")
print(f"  ND 95% CI: [{g_low_nd_lo:.3f}, {g_low_nd_hi:.3f}]")
print("\nHigh Noise (D>0.7) G-Score:")
print(f"  NT: {g_high_nt:.3f}")
print(f"  ND: {g_high_nd:.3f} -> ND is {((g_high_nd - g_high_nt)/g_high_nt)*100:.1f}% LESS rational (Meltdown).")
print("\nAttention Cost of High Social Pressure (S>0.8):")
print(f"  NT Lost Attention: {att_drop_nt:.3f}")
print(f"  ND Lost Attention: {att_drop_nd:.3f} -> Masking costs NDs {att_drop_nd/att_drop_nt:.1f}x more energy.")


# --- Plotting ---
sns.set_theme(style="whitegrid")

# Plot 1: G-Score Distribution (Violin Plot)
plt.figure(figsize=(10, 6))
sns.violinplot(data=df, x="Type", y="G")
plt.title("Distribution of Cognitive Vulnerability (G) by Neurotype")
plt.ylabel("Cognitive Vulnerability (G)")
plt.axhline(y=MELTDOWN_THRESHOLD_G, color="r", linestyle="--", label=f"Meltdown Threshold (G={MELTDOWN_THRESHOLD_G:.1f})")
plt.legend()
plt.savefig(FIGURES_DIR / "fig1_synthetic_g_distribution.png")
print(f"Saved {FIGURES_DIR / 'fig1_synthetic_g_distribution.png'}")

# Plot 2: Interaction of Noise and Rationality (Scatter with Fit)
plt.figure(figsize=(10, 6))
# Subsample for cleaner scatter plot
sample_df = df.sample(1000, random_state=RANDOM_SEED)
sns.scatterplot(data=sample_df, x="Environment_Noise", y="G", hue="Type", alpha=0.5, palette="muted")
sns.regplot(data=sample_df[sample_df['Type']=='Neurotypical'], x="Environment_Noise", y="G", scatter=False, color='b', label='NT Trend')
sns.regplot(data=sample_df[sample_df['Type']=='Autistic'], x="Environment_Noise", y="G", scatter=False, color='orange', label='ND Trend')
plt.title("Impact of Environmental Load on Cognitive Vulnerability")
plt.xlabel("Environmental Load (D)")
plt.ylabel("Cognitive Vulnerability (G)")
plt.legend()
plt.savefig(FIGURES_DIR / "fig2_noise_impact.png")
print(f"Saved {FIGURES_DIR / 'fig2_noise_impact.png'}")

