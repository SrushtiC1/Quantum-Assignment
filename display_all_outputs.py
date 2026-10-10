"""
================================================================================
CONSOLIDATED TERMINAL OUTPUT VIEWER
Displays all quantum circuit diagrams, resource tables, and simulation results.
================================================================================
"""

import sys
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import pandas as pd
from circuit_design import build_full_quantum_circuit

print("=" * 80)
print("  QUANTUM CIRCUIT DESIGN & ENTANGLEMENT ENGINE - LIVE TERMINAL REPORT")
print("=" * 80)

# ------------------------------------------------------------------------------
# 1. 2-QUBIT CIRCUIT DIAGRAMS (NON-ENTANGLED vs ENTANGLED)
# ------------------------------------------------------------------------------
print("\n" + "=" * 80)
print("SECTION 3.3: 2-QUBIT CIRCUIT ARCHITECTURES (ASCII DIAGRAMS)")
print("=" * 80)

qc_2q_non = build_full_quantum_circuit(num_qubits=2, depth=1, entangled=False)
print("\n[A] 2-Qubit Non-Entangled Circuit (Depth L=1, CX=0):")
print("-" * 60)
print(qc_2q_non.draw(output='text'))

qc_2q_cx = build_full_quantum_circuit(num_qubits=2, depth=1, entangled=True, entangler='cx')
print("\n[B] 2-Qubit Entangled CNOT Circuit (Depth L=1, CX=1):")
print("-" * 60)
print(qc_2q_cx.draw(output='text'))

qc_2q_cz = build_full_quantum_circuit(num_qubits=2, depth=1, entangled=True, entangler='cz')
print("\n[C] 2-Qubit Entangled CZ Circuit (Depth L=1, CZ=1):")
print("-" * 60)
print(qc_2q_cz.draw(output='text'))

qc_2q_d2 = build_full_quantum_circuit(num_qubits=2, depth=2, entangled=True, entangler='cx')
print("\n[D] 2-Qubit Entangled CNOT Circuit (Depth L=2, CX=2):")
print("-" * 60)
print(qc_2q_d2.draw(output='text'))


# ------------------------------------------------------------------------------
# 2. 4-QUBIT CIRCUIT DIAGRAMS (NON-ENTANGLED & ENTANGLED TOPOLOGIES)
# ------------------------------------------------------------------------------
print("\n" + "=" * 80)
print("SECTION 3.3: 4-QUBIT CIRCUIT ARCHITECTURES (FULL IRIS FEATURES)")
print("=" * 80)

qc_4q_non = build_full_quantum_circuit(num_qubits=4, depth=1, entangled=False)
print("\n[E] 4-Qubit Non-Entangled Circuit (Depth L=1, CX=0):")
print("-" * 60)
print(qc_4q_non.draw(output='text'))

qc_4q_linear = build_full_quantum_circuit(num_qubits=4, depth=1, entangled=True, entangler='cx', topology='linear')
print("\n[F] 4-Qubit Entangled CNOT Linear Topology (Depth L=1, CX=3):")
print("-" * 60)
print(qc_4q_linear.draw(output='text'))

qc_4q_circular = build_full_quantum_circuit(num_qubits=4, depth=1, entangled=True, entangler='cx', topology='circular')
print("\n[G] 4-Qubit Entangled CNOT Circular Topology (Depth L=1, CX=4):")
print("-" * 60)
print(qc_4q_circular.draw(output='text'))

qc_4q_full = build_full_quantum_circuit(num_qubits=4, depth=1, entangled=True, entangler='cx', topology='full')
print("\n[H] 4-Qubit Entangled CNOT All-to-All / Full Topology (Depth L=1, CX=6):")
print("-" * 60)
print(qc_4q_full.draw(output='text'))


# ------------------------------------------------------------------------------
# 3. RESOURCE AUDIT & PARAMETER TABLE (ALL 24 CONFIGURATIONS)
# ------------------------------------------------------------------------------
print("\n" + "=" * 80)
print("SECTION 3.4: QUANTUM RESOURCE & PARAMETER AUDIT TABLE")
print("=" * 80)

df_table = pd.read_csv("circuit_resource_table.csv")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', None)

summary_cols = [
    "Circuit_ID", "Qubits (n)", "Entanglement", "Entangler", "Topology",
    "Ansatz_Depth (L)", "Total_Circuit_Depth", "Trainable_Parameters (P)",
    "Single_Qubit_Gates", "Two_Qubit_Gates (2Q)", "Total_Comp_Gates"
]
print(df_table[summary_cols].to_string(index=False))


# ------------------------------------------------------------------------------
# 4. PERSON 3 SIMULATION BENCHMARK SUMMARY (IDEAL VS NOISY)
# ------------------------------------------------------------------------------
print("\n" + "=" * 80)
print("PERSON 3 SIMULATION SUMMARY: NOISE IMPACT ACROSS TIERS (N0 -> N3)")
print("=" * 80)

df_sim = pd.read_csv("person3_detailed_simulation_matrix.csv")

# Filter for Sample 0 across different noise tiers to show degradation
sample0 = df_sim[(df_sim["Sample_ID"] == 0) & (df_sim["Seed"] == 42)]
print("\n[Measurement Counts for Sample 0 (Seed 42, 1024 Shots)]:")
print(sample0[["Circuit_Type", "Noise_Tier", "Probability_State_00", "Raw_Counts"]].to_string(index=False))

print("\n" + "=" * 80)
print("ALL OUTPUTS SUCCESSFULLY RENDERED IN TERMINAL")
print("=" * 80)
