# ============================================================
# PERSON 3: QISKIT IMPLEMENTATION, NOISE MODELING & SIMULATION
# (POWERED BY PERSON 2 MODULAR QUANTUM CIRCUIT ARCHITECTURE)
# ============================================================

import sys
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError

# Import Person 2 modular circuit engine
from circuit_design import create_modular_classifier_circuit

print("=" * 70)
print("PERSON 3 - ADVANCED SIMULATION & NOISE MODELING ENGINE")
print("=" * 70)

# ============================================================
# 1. ENVIRONMENT & REPRODUCIBILITY SETUP (Report Section 4.1)
# ============================================================
SHOTS = 1024
SEEDS = [42, 123, 456, 789, 2026]  # Multi-seed robustness testing

print(f"[Config] Shots per job: {SHOTS}")
print(f"[Config] Random Seeds: {SEEDS}")

# ============================================================
# 2. DATA PREPROCESSING PIPELINE (Ensuring Environment Parity)
# ============================================================
iris = load_iris()
X = iris.data
y = iris.target

# Binary classification: Keep Setosa (0) and Versicolor (1)
# Select 2 features for 2-qubit or all 4 features for 4-qubit
mask = (y == 0) | (y == 1)
X_2q = X[mask][:, :2]
X_4q = X[mask]
y_bin = y[mask]

X_train_2q, X_test_2q, y_train, y_test = train_test_split(
    X_2q, y_bin, test_size=0.20, random_state=42, stratify=y_bin
)
X_train_4q, X_test_4q, _, _ = train_test_split(
    X_4q, y_bin, test_size=0.20, random_state=42, stratify=y_bin
)

scaler_2q = MinMaxScaler(feature_range=(0, np.pi))
X_train_q2 = np.clip(scaler_2q.fit_transform(X_train_2q), 0, np.pi)
X_test_q2 = np.clip(scaler_2q.transform(X_test_2q), 0, np.pi)

scaler_4q = MinMaxScaler(feature_range=(0, np.pi))
X_train_q4 = np.clip(scaler_4q.fit_transform(X_train_4q), 0, np.pi)
X_test_q4 = np.clip(scaler_4q.transform(X_test_4q), 0, np.pi)

print(f"[Data] Loaded {len(X_train_q2)} train samples and {len(X_test_q2)} test samples.")


# ============================================================
# 3. NOISE MODEL IMPLEMENTATION (Report Section 4.3)
# ============================================================
def build_noise_model(p_1q=0.0, p_2q=0.0, p_readout=0.0):
    """
    Constructs a Qiskit NoiseModel containing:
      - Single-qubit depolarizing error
      - Two-qubit depolarizing error (CX/CZ gates)
      - Classical readout measurement error
    """
    noise_model = NoiseModel()
    
    if p_1q > 0.0:
        error_1q = depolarizing_error(p_1q, 1)
        noise_model.add_all_qubit_quantum_error(error_1q, ['ry', 'x', 'h'])
        
    if p_2q > 0.0:
        error_2q = depolarizing_error(p_2q, 2)
        noise_model.add_all_qubit_quantum_error(error_2q, ['cx', 'cz'])
        
    if p_readout > 0.0:
        ro_error = ReadoutError([[1 - p_readout, p_readout], [p_readout, 1 - p_readout]])
        noise_model.add_all_qubit_readout_error(ro_error)
        
    return noise_model

# Defining Noise Tiers (N0 to N3)
noise_tiers = {
    "N0_Ideal": build_noise_model(0.0, 0.0, 0.0),
    "N1_Low_Noise": build_noise_model(p_1q=0.0005, p_2q=0.002, p_readout=0.005),
    "N2_Moderate_Noise": build_noise_model(p_1q=0.0020, p_2q=0.010, p_readout=0.020),
    "N3_High_Noise": build_noise_model(p_1q=0.0080, p_2q=0.035, p_readout=0.070)
}

# Attaching noise models to AerSimulator instances
simulators = {
    name: AerSimulator(noise_model=model) if name != "N0_Ideal" else AerSimulator()
    for name, model in noise_tiers.items()
}

print("\n[Noise Model] Successfully compiled and attached to AerSimulator tiers:")
for tier_name in simulators.keys():
    print(f" -> Active Simulator Backend: {tier_name}")


# ============================================================
# 4. EXECUTION PIPELINE ACROSS ALL TIERS & SEEDS
# ============================================================
def run_full_simulation_experiment():
    print("\n" + "=" * 50)
    print("RUNNING SIMULATION EXPERIMENT PIPELINE...")
    print("=" * 50)
    
    # 2-qubit weights: Depth 1 has (1+1)*2 = 4 parameters
    weights_2q = np.array([0.5, -0.5, 0.25, -0.25])
    experiment_records = []
    
    # Evaluate across test instances
    for idx, sample_features in enumerate(X_test_q2[:5]):
        # Person 2 modular circuit generator
        qc_non = create_modular_classifier_circuit(2, sample_features, weights_2q, depth=1, entangled=False)
        qc_ent = create_modular_classifier_circuit(2, sample_features, weights_2q, depth=1, entangled=True, entangler='cx')
        
        for circuit_type, qc in [("Non_Entangled", qc_non), ("Entangled_CNOT", qc_ent)]:
            for tier_name, backend in simulators.items():
                for seed in SEEDS:
                    transpiled_qc = transpile(qc, backend, optimization_level=1)
                    job = backend.run(transpiled_qc, shots=SHOTS, seed_simulator=seed)
                    result = job.result()
                    counts = result.get_counts()
                    
                    total_counts = sum(counts.values())
                    prob_0 = counts.get('00', 0) / total_counts
                    
                    experiment_records.append({
                        "Sample_ID": idx,
                        "Circuit_Type": circuit_type,
                        "Noise_Tier": tier_name,
                        "Seed": seed,
                        "Shots": SHOTS,
                        "Raw_Counts": str(counts),
                        "Probability_State_00": prob_0
                    })

    df_out = pd.DataFrame(experiment_records)
    df_out.to_csv("person3_detailed_simulation_matrix.csv", index=False)
    print("\n[SUCCESS] Simulation complete!")
    print("[SAVED] Output file generated: 'person3_detailed_simulation_matrix.csv'")
    print(df_out.head(10).to_string(index=False))

if __name__ == "__main__":
    run_full_simulation_experiment()
    print("\n" + "=" * 70)
    print("PERSON 3 TASK EXECUTION FINISHED SUCCESSFULLY")
    print("=" * 70)
