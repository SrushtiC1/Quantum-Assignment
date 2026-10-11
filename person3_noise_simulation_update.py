# ============================================================
# PERSON 3: BATCH EXPERIMENT RUNNER FOR NOISE & CLASSIFICATION
# ============================================================

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, readout_error

print("=" * 70)
print("PERSON 3 - BATCH EXPERIMENT RUNNER INITIALIZED")
print("=" * 70)

# --- 1. CONFIGURATION (Matching Project Table Specifications) ---
SHOTS = 1024
SEEDS = [42, 123, 456, 789, 2026]

# --- 2. NOISE MODEL GENERATOR ---
def build_noise_model(p_1q, p_2q, p_readout):
    noise_model = NoiseModel()
    if p_1q > 0.0:
        noise_model.add_all_quantum_error(depolarizing_error(p_1q, 1), ['ry', 'x', 'h'])
    if p_2q > 0.0:
        noise_model.add_all_quantum_error(depolarizing_error(p_2q, 2), ['cx'])
    if p_readout > 0.0:
        noise_model.add_all_readout_error(readout_error([
            [1 - p_readout, p_readout], 
            [p_readout, 1 - p_readout]
        ]))
    return noise_model

# --- 3. SIMULATOR TIERS (Ideal, Low, Moderate) ---
simulators = {
    "Ideal": AerSimulator(),
    "Low_Noise": AerSimulator(noise_model=build_noise_model(0.0005, 0.002, 0.005)),
    "Moderate_Noise": AerSimulator(noise_model=build_noise_model(0.0020, 0.010, 0.020))
}

# --- 4. DATA PREPARATION ---
def load_and_preprocess_data():
    iris = load_iris()
    X, y = iris.data, iris.target
    mask = (y == 0) | (y == 1)
    X = X[mask][:, :2]
    y = y[mask]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    return scaler.fit_transform(X_train), scaler.transform(X_test), y_train, y_test

# --- 5. CIRCUIT BUILDER (Entangled vs Non-Entangled Baseline) ---
def build_circuit(features, weights, entangled=False):
    qc = QuantumCircuit(2, 2)
    qc.ry(features[0], 0)
    qc.ry(features[1], 1)
    qc.ry(weights[0], 0)
    qc.ry(weights[1], 1)
    if entangled:
        qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    return qc

# --- 6. BATCH EXECUTION ENGINE ---
def execute_batch_simulations():
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    dummy_weights = np.array([0.4, -0.4])
    all_records = []
    
    print(f"\n[Execution] Starting batch simulations across {len(SEEDS)} seeds and {len(simulators)} tiers...")
    
    for sample_idx, sample in enumerate(X_test[:5]): # Testing subset for batch processing
        for circuit_name, is_entangled in [("Non_Entangled_Baseline", False), ("Entangled_Circuit", True)]:
            qc = build_circuit(sample, dummy_weights, entangled=is_entangled)
            
            for tier_name, backend in simulators.items():
                transpiled = transpile(qc, backend, optimization_level=1)
                
                for seed in SEEDS:
                    job = backend.run(transpiled, shots=SHOTS, seed_simulator=seed)
                    counts = job.result().get_counts()
                    
                    all_records.append({
                        "Sample_Index": sample_idx,
                        "Circuit_Architecture": circuit_name,
                        "Noise_Tier": tier_name,
                        "Random_Seed": seed,
                        "Shots": SHOTS,
                        "Counts": str(counts)
                    })
                    
    df_results = pd.DataFrame(all_records)
    output_filename = "person3_batch_experiment_matrix.csv"
    df_results.to_csv(output_filename, index=False)
    print(f"\n[SUCCESS] Batch execution complete! Output saved to '{output_filename}'.")

if __name__ == "__main__":
    execute_batch_simulations()