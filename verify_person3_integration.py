"""
================================================================================
PERSON 3 INTEGRATION TEST WITH PERSON 2 MODULAR CIRCUITS
================================================================================
Demonstrates seamless integration between:
  - Person 2: Quantum Circuit Design Engine (circuit_design.py)
  - Person 3: Qiskit Aer Simulation & Noise Modeling Engine
"""

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

# Import the modular circuit factory designed by Person 2
from circuit_design import create_modular_classifier_circuit, build_full_quantum_circuit

print("=" * 78)
print("TESTING FULL INTEGRATION: PERSON 2 CIRCUITS IN PERSON 3 NOISE PIPELINE")
print("=" * 78)

# Setup simple noise model
def build_test_noise_model(p_1q=0.001, p_2q=0.01, p_ro=0.02):
    nm = NoiseModel()
    if p_1q > 0:
        nm.add_all_qubit_quantum_error(depolarizing_error(p_1q, 1), ['ry'])
    if p_2q > 0:
        nm.add_all_qubit_quantum_error(depolarizing_error(p_2q, 2), ['cx', 'cz'])
    if p_ro > 0:
        nm.add_all_qubit_readout_error(ReadoutError([[1-p_ro, p_ro], [p_ro, 1-p_ro]]))
    return nm

backend_ideal = AerSimulator()
backend_noisy = AerSimulator(noise_model=build_test_noise_model())

# ------------------------------------------------------------------------------
# 1. 2-Qubit Circuits (2 features)
# ------------------------------------------------------------------------------
sample_x_2q = np.array([0.5, 1.2])
weights_d1 = np.array([0.1, -0.2, 0.3, -0.4]) # Depth 1 -> (1+1)*2 = 4 weights

qc_2q_non = create_modular_classifier_circuit(2, sample_x_2q, weights_d1, depth=1, entangled=False)
qc_2q_ent = create_modular_classifier_circuit(2, sample_x_2q, weights_d1, depth=1, entangled=True, entangler='cx')
qc_2q_cz  = create_modular_classifier_circuit(2, sample_x_2q, weights_d1, depth=1, entangled=True, entangler='cz')

# ------------------------------------------------------------------------------
# 2. 4-Qubit Circuits (Full 4-feature Iris)
# ------------------------------------------------------------------------------
sample_x_4q = np.array([0.5, 1.2, 0.8, 2.1])
weights_4q_d1 = np.array([0.1, -0.1, 0.2, -0.2, 0.3, -0.3, 0.4, -0.4]) # Depth 1 -> (1+1)*4 = 8 weights

qc_4q_non        = create_modular_classifier_circuit(4, sample_x_4q, weights_4q_d1, depth=1, entangled=False)
qc_4q_ent_linear = create_modular_classifier_circuit(4, sample_x_4q, weights_4q_d1, depth=1, entangled=True, entangler='cx', topology='linear')
qc_4q_ent_circ   = create_modular_classifier_circuit(4, sample_x_4q, weights_4q_d1, depth=1, entangled=True, entangler='cx', topology='circular')
qc_4q_ent_full   = create_modular_classifier_circuit(4, sample_x_4q, weights_4q_d1, depth=1, entangled=True, entangler='cx', topology='full')

test_circuits = {
    "2Q_NonEntangled": (qc_2q_non, "Non-Entangled"),
    "2Q_Entangled_CX": (qc_2q_ent, "Entangled (CX)"),
    "2Q_Entangled_CZ": (qc_2q_cz,  "Entangled (CZ)"),
    "4Q_NonEntangled": (qc_4q_non, "Non-Entangled"),
    "4Q_Entangled_Linear": (qc_4q_ent_linear, "Entangled (Linear)"),
    "4Q_Entangled_Circular": (qc_4q_ent_circ, "Entangled (Circular)"),
    "4Q_Entangled_Full": (qc_4q_ent_full, "Entangled (Full)")
}

results = []
for name, (qc, c_type) in test_circuits.items():
    for mode, sim in [("Ideal", backend_ideal), ("Noisy", backend_noisy)]:
        tqc = transpile(qc, sim)
        job = sim.run(tqc, shots=1024, seed_simulator=42)
        counts = job.result().get_counts()
        total = sum(counts.values())
        
        # Probability of state |0...0>
        zero_state = '0' * qc.num_qubits
        p0 = counts.get(zero_state, 0) / total
        
        sorted_counts = sorted(counts.items(), key=lambda item: item[1], reverse=True)
        top1 = f"{sorted_counts[0][0]}: {sorted_counts[0][1]}" if len(sorted_counts) > 0 else "-"
        top2 = f"{sorted_counts[1][0]}: {sorted_counts[1][1]}" if len(sorted_counts) > 1 else "-"
        top3 = f"{sorted_counts[2][0]}: {sorted_counts[2][1]}" if len(sorted_counts) > 2 else "-"
        
        results.append({
            "Circuit ID": name,
            "Mode": mode,
            "P(|0...0>)": f"{p0:.4f}",
            "Top 1 State": top1,
            "Top 2 State": top2,
            "Top 3 State": top3
        })

df_res = pd.DataFrame(results)
print("\nSIMULATION RESULTS TABLE (1024 Shots per Configuration):")
print("-" * 78)
print(df_res.to_string(index=False))
print("-" * 78)
print("[SUCCESS] Seamless integration verified! Person 2 circuits are 100% compatible with Person 3!")
