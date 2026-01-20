
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
# --- Configuration ---
DATA_PATH = '../data/extracted/BIG5/data.csv'
OUTPUT_REPORT = '../manuscript/Real_World_Validation_Report.md'
FIGURE_PATH = '../figures/'

print("Loading Big Five Data...")
# Load data (handling potential delimiter issues)
try:
    df = pd.read_csv(DATA_PATH, sep='\t') # Often tab separated
    if df.shape[1] < 5:
        df = pd.read_csv(DATA_PATH, sep=',') # Try comma
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

print(f"Dataset Size: {len(df)} participants")

# --- Preprocessing & normalization ---
# Scores in this dataset are usually raw sums (e.g. 10-50). We normalize to 0-1.
# Columns: E1..E10, N1..N10, etc. 
# But usually there are already aggregate scores if we look at the codebook.
# Let's check headers first. If not aggregate, we sum.
# Inspecting columns... assuming standard structure:
# 'E' = Extraversion, 'N' = Neuroticism, 'A' = Agreeableness, 'C' = Conscientiousness, 'O' = Openness

# Calculate Trait Scores (Mean of items, normalized 1-5 -> 0-1)
traits = ['E', 'N', 'A', 'C', 'O']
for t in traits:
    cols = [c for c in df.columns if c.startswith(t) and c[1:].isdigit()]
    if cols:
        # 0 = nothing, 1-5 scale. Remove 0s.
        df[cols] = df[cols].replace(0, np.nan)
        # Normalize (Sum / Count - 1) / 4 -> [0, 1]
        df[f'{t}_norm'] = (df[cols].mean(axis=1) - 1) / 4

# Drop rows with NaNs
df = df.dropna(subset=[f'{t}_norm' for t in traits])
print(f"Cleaned Dataset: {len(df)} participants")

# --- G-Model Mapping ---
# Constants
I_sim = np.random.normal(1.0, 0.15, len(df)) # Simulated IQ (correlated with Openness slightly?)
I_sim = np.clip(I_sim, 0.5, 1.5)
S_env = 0.8 # High Social Pressure Environment
D_env = 0.8 # High Digital Noise Environment

# Variables derived from Big Five
# A (Attention) = Conscientiousness
df['A_raw'] = df['C_norm'] * 1.0 # scaling
# E (Emotional Reg) = Inverse Neuroticism
df['E_reg'] = 1.0 - df['N_norm'] 
# B_mot (Motivated Bias) = Inverse Openness (Dogmatism)
df['B_mot'] = (1.0 - df['O_norm']) * 0.8 
# C (Critical Thinking) = Openness + Intelligence (proxy)
df['Crit'] = (df['O_norm'] + (I_sim - 1.0)) / 2 + 0.5

# --- Definition of Phenotypes ---

# 1. Neurotypical (NT): High Extraversion, Low Neuroticism
df['Is_NT'] = (df['E_norm'] > 0.6) & (df['N_norm'] < 0.4)

# 2. Monotropic / Autistic-Like (ND): Low Extraversion, High Conscientiousness, High Neuroticism (Sensitivity)
# Note: "High Neuroticism" in Big 5 often maps to "Sensory Sensitivity" in ASD context.
df['Is_ND'] = (df['E_norm'] < 0.4) & (df['C_norm'] > 0.6) & (df['N_norm'] > 0.6)


# print(f" identified {df['Is_NT'].sum()} Neurotypicals and {df['Is_ND'].sum()} Monotropic Profiles")


# --- G-Calculation Function ---
def calculate_g(row, agent_type):
    # Base Params
    I = I_sim[row.name] if row.name < len(I_sim) else 1.0
    A = row['A_raw']
    E = row['E_reg']
    B_mot = row['B_mot']
    Crit = row['Crit']
    
    # 1. Processing Error
    term_1 = 0.2 / I # assume consant B_err for simplicity
    
    # 2. Motivated Bias & Masking
    if agent_type == 'ND':
        # ND: Low B_mot (resistance to social bias)
        B_actual = B_mot * 0.5 
        # ND: High Masking Cost
        masking_cost = 0.5 * S_env
    else:
        # NT: Normal B_mot
        B_actual = B_mot * (1 + 0.5 * S_env)
        # NT: Low Masking Cost
        masking_cost = 0.05 * S_env
        
    # 3. Environmental Load
    A_eff = max(0.01, A - masking_cost)
    
    if agent_type == 'ND':
        # ND: Hyper-sensitivity to Noise (D_eff scales exp)
        D_eff = D_env * np.exp(max(0, D_env - 0.4))
    else:
        # NT: Standard filtering
        D_eff = D_env * np.exp(max(0, D_env - 0.7))
        
    term_2 = D_eff / A_eff
    
    # 4. Social Reg
    term_3 = (S_env * (1 - Crit)) / max(0.1, E)
    
    G = (0.3 * (term_1 + B_actual)) + (0.5 * term_2) + (0.2 * term_3)
    return G


# --- Full Dataset Analysis ---
print(f"Processing FULL Dataset: {len(df)} participants...")

# Instead of filtering, we calculate G for EVERYONE under both models, 
# then assign a 'Neurotype Probability' to weigh the results.

# 1. Calculate Monotropic Score (0.0 to 1.0) for every participant
# Based on: High C, High N, Low E
# Score = (C + N + (1-E)) / 3
df['Monotropism_Score'] = (df['C_norm'] + df['N_norm'] + (1.0 - df['E_norm'])) / 3.0

# 2. Define Groups based on Score Quantiles (Top 20% vs Bottom 20%)
# This ensures we use the full distribution relative to the population
q_high = df['Monotropism_Score'].quantile(0.80)
q_low = df['Monotropism_Score'].quantile(0.20)

df['Group'] = 'Mixed'
df.loc[df['Monotropism_Score'] >= q_high, 'Group'] = 'Monotropic_High'
df.loc[df['Monotropism_Score'] <= q_low, 'Group'] = 'Neurotypical_High'

print(f"Group Split:")
print(df['Group'].value_counts())

# 3. Calculate G-Scores
# For Monotropic_High, we use the ND model logic.
# For Neurotypical_High, we use the NT model logic.
# For Mixed, we calculate an average (or exclude from extreme comparison).

def calculate_g_dynamic(row):
    # Determine model parameters based on individual traits
    # This effectively runs the simulation for the specific person
    
    agent_type = 'NT'
    if row['Group'] == 'Monotropic_High':
        agent_type = 'ND'
    
    # Base Params
    I = I_sim[row.name] if row.name < len(I_sim) else 1.0
    A = row['A_raw']
    E = row['E_reg']
    B_mot = row['B_mot']
    Crit = row['Crit']
    
    # 1. Processing Error
    term_1 = 0.2 / I
    
    # 2. Motivated Bias & Masking
    if agent_type == 'ND':
        B_actual = B_mot * 0.5 
        masking_cost = 0.5 * S_env
    else: # NT or Mixed
        B_actual = B_mot * (1 + 0.5 * S_env)
        masking_cost = 0.05 * S_env
        
    # 3. Environmental Load
    A_eff = max(0.01, A - masking_cost)
    
    if agent_type == 'ND':
        D_eff = D_env * np.exp(max(0, D_env - 0.4))
    else:
        D_eff = D_env * np.exp(max(0, D_env - 0.7))
        
    term_2 = D_eff / A_eff
    term_3 = (S_env * (1 - Crit)) / max(0.1, E)
    G = (0.3 * (term_1 + B_actual)) + (0.5 * term_2) + (0.2 * term_3)
    return G

df['G_Score'] = df.apply(calculate_g_dynamic, axis=1)
output_df = df[df['Group'] != 'Mixed'].copy() # Focus on the clear phenotypes for the paper

# Statistics
stats = output_df.groupby('Group')['G_Score'].describe()
print("\n--- Comparative Results ---")
print(stats)

# Save Report

# Old validation block removed to avoid key error


# --- Plotting ---
sns.set_theme(style="whitegrid")


# Plot 3: Real World G-Score Distribution (KDE)
plt.figure(figsize=(10, 6))
sns.kdeplot(data=output_df, x="G_Score", hue="Group", fill=True, palette="rocket", common_norm=False)
plt.axvline(x=2.0, color='r', linestyle='--', label='Meltdown Threshold (G=2.0)')
plt.title("Real-World Density of Cognitive Collapse (Big Five Dataset)")
plt.xlabel("Cognitive Vulnerability (G)")
plt.legend(title='Phenotype')
plt.legend(title='Phenotype')
plt.savefig(f"{FIGURE_PATH}fig3_realworld_density.png")
print(f"Saved {FIGURE_PATH}fig3_realworld_density.png")

# Plot 4: Meltdown Rates Bar Chart
failure_rates = output_df.groupby('Group')['G_Score'].apply(lambda x: (x > 2.0).mean() * 100).reset_index()
failure_rates.columns = ['Group', 'Meltdown_Rate']

plt.figure(figsize=(8, 6))
sns.barplot(data=failure_rates, x="Group", y="Meltdown_Rate", palette="rocket")
plt.title("Singularity Rate (% of Population in Meltdown)")
plt.ylabel("Percent > Critical Threshold")
plt.ylim(0, 100)
for index, row in failure_rates.iterrows():
    plt.text(index, row.Meltdown_Rate + 2, f"{row.Meltdown_Rate:.1f}%", color='black', ha="center")
for index, row in failure_rates.iterrows():
    plt.text(index, row.Meltdown_Rate + 2, f"{row.Meltdown_Rate:.1f}%", color='black', ha="center")
plt.savefig(f"{FIGURE_PATH}fig4_meltdown_rates.png")
print(f"Saved {FIGURE_PATH}fig4_meltdown_rates.png")

# Update Summary Text for Report
stats_nt = stats.loc['Neurotypical_High']
stats_nd = stats.loc['Monotropic_High']

with open(OUTPUT_REPORT, 'w') as f:
    f.write("# Demographic Risk Projection Report (Big Five Dataset)\n\n")
    f.write(f"**Dataset**: Open Psychometrics Big Five (N={len(df)})\n")
    f.write(f"**Method**: Monotropism Score Quantiles (Top 20% vs Bottom 20%)\n")
    f.write("**Note**: This report represents a theoretical stress-test of the population, not clinical diagnosis.\n\n")
    f.write("## 1. Population Statistics\n")
    f.write(f"- Neurotypical (Poly) Count: {len(df[df['Group']=='Neurotypical_High'])}\n")
    f.write(f"- Monotropic (ND) Count: {len(df[df['Group']=='Monotropic_High'])}\n\n")
    f.write("## 2. Rationality Scores (G) under Stress (D=0.8, S=0.8)\n")
    f.write(stats.to_string())
    f.write("\n\n## 3. Projected Findings\n")
    f.write(f"- The Monotropic group shows a Mean G of **{stats_nd['mean']:.2f}** vs NT **{stats_nt['mean']:.2f}**.\n")
    
    fr_nd = failure_rates[failure_rates['Group']=='Monotropic_High']['Meltdown_Rate'].values[0]
    fr_nt = failure_rates[failure_rates['Group']=='Neurotypical_High']['Meltdown_Rate'].values[0]
    
    f.write(f"- **Theoretical Singularity Rate (Risk > 2.0)**:\n")
    f.write(f"  - ND: {fr_nd:.1f}%\n")
    f.write(f"  - NT: {fr_nt:.1f}%\n")

print(f"Report saved to {OUTPUT_REPORT}")
