# ============================================================
# PERSON 3: QISKIT IMPLEMENTATION, NOISE MODELING & SIMULATION
# ============================================================

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, readout_error

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

# Binary classification: Keep Setosa (0) and Versicolor (1) and select 2 features
mask = (y == 0) | (y == 1)
X = X[mask][:, :2]
y = y[mask]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

angle_scaler = MinMaxScaler(feature_range=(0, np.pi))
X_train_q = np.clip(angle_scaler.fit_transform(X_train), 0, np.pi)
X_test_q = np.clip(angle_scaler.transform(X_test), 0, np.pi)

print(f"[Data] Training samples loaded: {X_train_q.shape[0]}")


# ============================================================
# 3. NOISE MODEL IMPLEMENTATION (Report Section 4.3)
# ============================================================
def build_noise_model(p_1q=0.0, p_2q=0.0, p_readout=0.0):
    """
    Constructs a Qiskit NoiseModel containing:
      - Single-qubit depolarizing error
      - Two-qubit depolarizing error (CX gates)
      - Classical readout measurement error
    """
    noise_model = NoiseModel()
    
    if p_1q > 0.0:
        error_1q = depolarizing_error(p_1q, 1)
        noise_model.add_all_quantum_error(error_1q, ['ry', 'x', 'h'])
        
    if p_2q > 0.0:
        error_2q = depolarizing_error(p_2q, 2)
        noise_model.add_all_quantum_error(error_2q, ['cx'])
        
    if p_readout > 0.0:
        ro_error = readout_error([[1 - p_readout, p_readout], [p_readout, 1 - p_readout]])
        noise_model.add_all_readout_error(ro_error)
        
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
# 4. QUANTUM CIRCUIT DEFINITIONS (Non-Entangled vs Entangled)
# ============================================================
def create_classifier_circuit(features, weights, entangled=False):
    qc = QuantumCircuit(2, 2)
    
    # Feature map (Data Encoding)
    qc.ry(features[0], 0)
    qc.ry(features[1], 1)
    
    # Ansatz (Trainable Layers)
    qc.ry(weights[0], 0)
    qc.ry(weights[1], 1)
    
    # Entanglement Layer
    if entangled:
        qc.cx(0, 1)
        
    qc.measure([0, 1], [0, 1])
    return qc


# ============================================================
# 5. EXECUTION PIPELINE ACROSS ALL TIERS & SEEDS
# ============================================================
def run_full_simulation_experiment():
    print("\n" + "=" * 50)
    print("RUNNING SIMULATION EXPERIMENT PIPELINE...")
    print("=" * 50)
    
    dummy_weights = np.array([0.5, -0.5]) # Representative parameter values for testing
    experiment_records = []
    
    # Run test loop across first few test instances for demonstration
    for idx, sample_features in enumerate(X_test_q[:5]):
        
        # Build circuits
        qc_non = create_classifier_circuit(sample_features, dummy_weights, entangled=False)
        qc_ent = create_classifier_circuit(sample_features, dummy_weights, entangled=True)
        
        for circuit_type, qc in [("Non_Entangled", qc_non), ("Entangled", qc_ent)]:
            for tier_name, backend in simulators.items():
                for seed in SEEDS:
                    
                    # Transpile and run simulation
                    transpiled_qc = transpile(qc, backend, optimization_level=1)
                    job = backend.run(transpiled_qc, shots=SHOTS, seed_simulator=seed)
                    result = job.result()
                    counts = result.get_counts()
                    
                    # Calculate estimated probability of state '00' or '11' for preliminary metrics
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