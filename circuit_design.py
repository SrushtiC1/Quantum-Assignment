"""
================================================================================
QUANTUM CIRCUIT DESIGN & ENTANGLEMENT ARCHITECTURE ENGINE
Author: Quantum Circuit Architect & Entanglement Specialist
Role: Person 2 - Entanglement & Quantum Circuit Design
================================================================================

This module implements:
  1. Quantum Feature Maps (Angle Encoding, ZZFeatureMap)
  2. Non-Entangled Parameterized Ansatz (Layers L=1, 2, 3)
  3. Entangled Parameterized Ansatz (CNOT / CZ with Linear, Circular, All-to-all topologies)
  4. 2-Qubit & 4-Qubit Circuit Generators
  5. Exact Resource Profiler (Gate counts, 2Q gate counts, Depth, Parameter counts)
  6. High-Resolution Circuit Diagram Exporter (PNG & ASCII)
  7. Parity Adapter for Person 3 Noise Simulation Engine
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure UTF-8 output on Windows console
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector, Parameter
from qiskit.circuit.library import ZZFeatureMap

# Ensure output directory for figures
OUTPUT_DIR = "circuit_diagrams"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==============================================================================
# 1. QUANTUM FEATURE MAPS (DATA ENCODING)
# ==============================================================================

def build_angle_feature_map(num_qubits: int, rotation: str = 'Y') -> QuantumCircuit:
    """
    Constructs an Angle Encoding Feature Map.
    Each classical feature x_i is mapped to the rotation angle of qubit i:
        |psi(x)> = tensor_{i=0}^{n-1} R_k(x_i) |0>
    where k in {X, Y, Z}. Standard is R_Y to keep state amplitudes real.
    """
    qc = QuantumCircuit(num_qubits, name=f"AngleMap_{rotation}")
    x_params = ParameterVector('x', num_qubits)
    
    for i in range(num_qubits):
        if rotation.upper() == 'Y':
            qc.ry(x_params[i], i)
        elif rotation.upper() == 'Z':
            qc.rz(x_params[i], i)
        elif rotation.upper() == 'X':
            qc.rx(x_params[i], i)
        else:
            raise ValueError(f"Unsupported rotation axis: {rotation}")
            
    return qc


def build_zz_feature_map(num_qubits: int, reps: int = 1) -> QuantumCircuit:
    """
    Constructs a second-order Pauli-Z (ZZ) Feature Map.
    Encodes both individual features and non-linear pairwise interactions:
        U_Phi(x) = exp( i sum_j x_j Z_j + sum_{j<k} (pi - x_j)(pi - x_k) Z_j Z_k )
    Uses Hadamard gates + RZ + CNOT pairs.
    """
    return ZZFeatureMap(feature_dimension=num_qubits, reps=reps, entanglement='linear')


# ==============================================================================
# 2. VARIATIONAL ANSATZ BUILDERS (NON-ENTANGLED & ENTANGLED)
# ==============================================================================

def get_entanglement_pairs(num_qubits: int, topology: str = 'linear'):
    """
    Generates qubit pairs for 2-qubit entangling gates based on topology.
    Topologies:
      - 'linear': (0,1), (1,2), ..., (n-2, n-1)
      - 'circular': linear + (n-1, 0) (for n >= 3)
      - 'full': all combinations (i, j) with i < j
    """
    pairs = []
    if topology == 'linear':
        for i in range(num_qubits - 1):
            pairs.append((i, i + 1))
    elif topology == 'circular':
        for i in range(num_qubits - 1):
            pairs.append((i, i + 1))
        if num_qubits > 2:
            pairs.append((num_qubits - 1, 0))
    elif topology == 'full':
        for i in range(num_qubits):
            for j in range(i + 1, num_qubits):
                pairs.append((i, j))
    else:
        raise ValueError(f"Unknown topology: {topology}")
    return pairs


def build_ansatz(
    num_qubits: int,
    depth: int = 1,
    entangled: bool = False,
    entangler: str = 'cx',
    topology: str = 'linear',
    rotations: str = 'ry'
) -> QuantumCircuit:
    """
    Builds a Parameterized Quantum Ansatz.
    
    Structure:
      - For each layer l in {1, ..., depth}:
          - Parameterized single-qubit rotation layer: R_Y(theta_{l, i})
          - If entangled:
              - 2-qubit entangling layer (CX or CZ) across specified topology
      - Final parameterized rotation layer: R_Y(theta_{final, i})
    
    Total parameters:
      - If single rotation (e.g. 'ry'): (depth + 1) * num_qubits
      - If dual rotation (e.g. 'ry_rz'): 2 * (depth + 1) * num_qubits
    """
    name_prefix = f"Ansatz_{'Ent' if entangled else 'NonEnt'}_D{depth}_{entangler}_{topology}"
    qc = QuantumCircuit(num_qubits, name=name_prefix)
    
    # Calculate parameter count
    rot_count = 2 if rotations == 'ry_rz' else 1
    total_params = (depth + 1) * num_qubits * rot_count
    theta = ParameterVector('theta', total_params)
    param_idx = 0
    
    pairs = get_entanglement_pairs(num_qubits, topology) if entangled else []
    
    for l in range(depth):
        # 1. Parameterized single-qubit rotations
        for i in range(num_qubits):
            if rotations == 'ry':
                qc.ry(theta[param_idx], i)
                param_idx += 1
            elif rotations == 'ry_rz':
                qc.ry(theta[param_idx], i)
                param_idx += 1
                qc.rz(theta[param_idx], i)
                param_idx += 1
                
        # 2. Entanglement layer (if enabled)
        if entangled and len(pairs) > 0:
            for c, t in pairs:
                if entangler.lower() == 'cx':
                    qc.cx(c, t)
                elif entangler.lower() == 'cz':
                    qc.cz(c, t)
                else:
                    raise ValueError(f"Unsupported entangler: {entangler}")
        
        qc.barrier()
        
    # Final rotation layer
    for i in range(num_qubits):
        if rotations == 'ry':
            qc.ry(theta[param_idx], i)
            param_idx += 1
        elif rotations == 'ry_rz':
            qc.ry(theta[param_idx], i)
            param_idx += 1
            qc.rz(theta[param_idx], i)
            param_idx += 1
            
    return qc


# ==============================================================================
# 3. FULL END-TO-END QUANTUM CLASSIFIER CIRCUIT BUILDER
# ==============================================================================

def build_full_quantum_circuit(
    num_qubits: int = 2,
    depth: int = 1,
    entangled: bool = False,
    entangler: str = 'cx',
    topology: str = 'linear',
    feature_map_type: str = 'angle',
    include_measurement: bool = True
) -> QuantumCircuit:
    """
    Assembles the complete end-to-end circuit:
        |0> --- [ Feature Map ] === [ Parameterized Ansatz ] === [ Measurement ]
    """
    qc = QuantumCircuit(num_qubits, num_qubits)
    
    # Feature Map
    if feature_map_type == 'angle':
        fmap = build_angle_feature_map(num_qubits, rotation='Y')
    elif feature_map_type == 'zz':
        fmap = build_zz_feature_map(num_qubits, reps=1)
    else:
        raise ValueError(f"Unknown feature map type: {feature_map_type}")
        
    qc.compose(fmap, inplace=True)
    qc.barrier()
    
    # Ansatz
    ansatz = build_ansatz(
        num_qubits=num_qubits,
        depth=depth,
        entangled=entangled,
        entangler=entangler,
        topology=topology,
        rotations='ry'
    )
    qc.compose(ansatz, inplace=True)
    qc.barrier()
    
    # Measurement
    if include_measurement:
        qc.measure(range(num_qubits), range(num_qubits))
        
    return qc


# ==============================================================================
# 4. RESOURCE PROFILER (GATE COUNTS, 2Q GATES, DEPTH, PARAMETERS)
# ==============================================================================

def analyze_circuit_resources(qc: QuantumCircuit) -> dict:
    """
    Performs comprehensive resource profiling of a QuantumCircuit:
      - Circuit Depth
      - Total Gate Count
      - 1-Qubit Gate Count
      - 2-Qubit Gate Count
      - Breakdown by Gate Type (ry, rz, cx, cz, measure, barrier)
      - Number of Classical Parameters
    """
    ops = qc.count_ops()
    
    # 2-qubit gates
    two_q_gates = ops.get('cx', 0) + ops.get('cz', 0) + ops.get('swap', 0)
    
    # 1-qubit gates (excluding measurement and barriers)
    one_q_keys = [k for k in ops.keys() if k not in ['cx', 'cz', 'swap', 'measure', 'barrier']]
    one_q_gates = sum(ops[k] for k in one_q_keys)
    
    # Total computational gates (excluding barrier)
    total_comp_gates = one_q_gates + two_q_gates
    
    return {
        "Num_Qubits": qc.num_qubits,
        "Depth": qc.depth(),
        "Total_Comp_Gates": total_comp_gates,
        "Single_Qubit_Gates": one_q_gates,
        "Two_Qubit_Gates": two_q_gates,
        "Num_Parameters": qc.num_parameters,
        "Op_Counts": dict(ops)
    }


# ==============================================================================
# 5. BATCH EXPERIMENTATION & AUDIT GENERATION
# ==============================================================================

def run_circuit_resource_audit():
    """
    Generates a full matrix of configurations for both 2-qubit and 4-qubit
    architectures across non-entangled and entangled variants at depths L=1, 2, 3.
    Exports results to CSV and Markdown.
    """
    records = []
    
    configurations = [
        # (num_qubits, entangled, entangler, topology, depth, label)
        # --- 2-QUBIT ARCHITECTURES ---
        (2, False, 'none', 'none', 1, "2Q Non-Entangled (L=1)"),
        (2, False, 'none', 'none', 2, "2Q Non-Entangled (L=2)"),
        (2, False, 'none', 'none', 3, "2Q Non-Entangled (L=3)"),
        (2, True,  'cx',   'linear', 1, "2Q Entangled CNOT (L=1)"),
        (2, True,  'cx',   'linear', 2, "2Q Entangled CNOT (L=2)"),
        (2, True,  'cx',   'linear', 3, "2Q Entangled CNOT (L=3)"),
        (2, True,  'cz',   'linear', 1, "2Q Entangled CZ (L=1)"),
        (2, True,  'cz',   'linear', 2, "2Q Entangled CZ (L=2)"),
        (2, True,  'cz',   'linear', 3, "2Q Entangled CZ (L=3)"),
        
        # --- 4-QUBIT ARCHITECTURES ---
        (4, False, 'none', 'none', 1, "4Q Non-Entangled (L=1)"),
        (4, False, 'none', 'none', 2, "4Q Non-Entangled (L=2)"),
        (4, False, 'none', 'none', 3, "4Q Non-Entangled (L=3)"),
        (4, True,  'cx',   'linear', 1, "4Q Entangled CNOT Linear (L=1)"),
        (4, True,  'cx',   'linear', 2, "4Q Entangled CNOT Linear (L=2)"),
        (4, True,  'cx',   'linear', 3, "4Q Entangled CNOT Linear (L=3)"),
        (4, True,  'cx',   'circular', 1, "4Q Entangled CNOT Circular (L=1)"),
        (4, True,  'cx',   'circular', 2, "4Q Entangled CNOT Circular (L=2)"),
        (4, True,  'cx',   'circular', 3, "4Q Entangled CNOT Circular (L=3)"),
        (4, True,  'cx',   'full', 1, "4Q Entangled CNOT Full (L=1)"),
        (4, True,  'cx',   'full', 2, "4Q Entangled CNOT Full (L=2)"),
        (4, True,  'cx',   'full', 3, "4Q Entangled CNOT Full (L=3)"),
        (4, True,  'cz',   'linear', 1, "4Q Entangled CZ Linear (L=1)"),
        (4, True,  'cz',   'linear', 2, "4Q Entangled CZ Linear (L=2)"),
        (4, True,  'cz',   'linear', 3, "4Q Entangled CZ Linear (L=3)")
    ]
    
    for n_q, ent, ent_type, topo, d, desc in configurations:
        qc = build_full_quantum_circuit(
            num_qubits=n_q,
            depth=d,
            entangled=ent,
            entangler=ent_type if ent else 'cx',
            topology=topo if ent else 'linear',
            feature_map_type='angle',
            include_measurement=True
        )
        
        # Also profile ansatz separately
        ansatz_only = build_ansatz(
            num_qubits=n_q,
            depth=d,
            entangled=ent,
            entangler=ent_type if ent else 'cx',
            topology=topo if ent else 'linear'
        )
        
        metrics = analyze_circuit_resources(qc)
        ansatz_metrics = analyze_circuit_resources(ansatz_only)
        
        # Calculate classical trainable parameters (excluding feature map inputs)
        trainable_params = ansatz_only.num_parameters
        feature_params = n_q  # angle encoding has n_q features
        
        records.append({
            "Circuit_ID": desc,
            "Qubits (n)": n_q,
            "Entanglement": "Entangled" if ent else "Non-Entangled",
            "Entangler": ent_type.upper() if ent else "None",
            "Topology": topo.capitalize() if ent else "None",
            "Ansatz_Depth (L)": d,
            "Total_Circuit_Depth": metrics["Depth"],
            "Trainable_Parameters (P)": trainable_params,
            "Feature_Parameters": feature_params,
            "Total_Parameters": metrics["Num_Parameters"],
            "Single_Qubit_Gates": metrics["Single_Qubit_Gates"],
            "Two_Qubit_Gates (2Q)": metrics["Two_Qubit_Gates"],
            "Total_Comp_Gates": metrics["Total_Comp_Gates"],
            "Measurements": n_q
        })
        
    df = pd.DataFrame(records)
    csv_path = "circuit_resource_table.csv"
    df.to_csv(csv_path, index=False)
    
    md_path = "circuit_resource_table.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Section 3.4: Quantum Circuit Parameter & Resource Audit Table\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n")
        
    print(f"[SUCCESS] Resource Audit Complete! Saved to '{csv_path}' and '{md_path}'")
    return df


# ==============================================================================
# 6. CIRCUIT DIAGRAM EXPORTER (MATPLOTLIB & TEXT)
# ==============================================================================

def export_circuit_diagrams():
    """
    Renders and saves high-resolution publication-quality PNG diagrams
    and ASCII representations for key circuit configurations.
    """
    representative_configs = [
        # 2-Qubit
        (2, False, 'none', 'none', 1, "2q_non_entangled_depth1", "2-Qubit Non-Entangled (Depth L=1)"),
        (2, False, 'none', 'none', 2, "2q_non_entangled_depth2", "2-Qubit Non-Entangled (Depth L=2)"),
        (2, True,  'cx',   'linear', 1, "2q_entangled_cnot_depth1", "2-Qubit Entangled CNOT (Depth L=1)"),
        (2, True,  'cx',   'linear', 2, "2q_entangled_cnot_depth2", "2-Qubit Entangled CNOT (Depth L=2)"),
        (2, True,  'cz',   'linear', 1, "2q_entangled_cz_depth1", "2-Qubit Entangled CZ (Depth L=1)"),
        
        # 4-Qubit
        (4, False, 'none', 'none', 1, "4q_non_entangled_depth1", "4-Qubit Non-Entangled (Depth L=1)"),
        (4, False, 'none', 'none', 2, "4q_non_entangled_depth2", "4-Qubit Non-Entangled (Depth L=2)"),
        (4, True,  'cx',   'linear', 1, "4q_entangled_cnot_linear_depth1", "4-Qubit Entangled CNOT Linear (Depth L=1)"),
        (4, True,  'cx',   'linear', 2, "4q_entangled_cnot_linear_depth2", "4-Qubit Entangled CNOT Linear (Depth L=2)"),
        (4, True,  'cx',   'circular', 1, "4q_entangled_cnot_circular_depth1", "4-Qubit Entangled CNOT Circular (Depth L=1)"),
        (4, True,  'cx',   'full', 1, "4q_entangled_cnot_full_depth1", "4-Qubit Entangled CNOT Full (Depth L=1)"),
        (4, True,  'cz',   'linear', 1, "4q_entangled_cz_linear_depth1", "4-Qubit Entangled CZ Linear (Depth L=1)")
    ]
    
    text_diagrams = []
    
    for n_q, ent, ent_type, topo, d, fname, title in representative_configs:
        qc = build_full_quantum_circuit(
            num_qubits=n_q,
            depth=d,
            entangled=ent,
            entangler=ent_type if ent else 'cx',
            topology=topo if ent else 'linear',
            feature_map_type='angle',
            include_measurement=True
        )
        
        # 1. Save graphical PNG using matplotlib
        img_path = os.path.join(OUTPUT_DIR, f"{fname}.png")
        try:
            fig = qc.draw(output='mpl', style='iqp')
            fig.suptitle(title, fontsize=12, fontweight='bold', y=0.98)
            plt.tight_layout()
            fig.savefig(img_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f" -> Generated PNG: {img_path}")
        except Exception as e:
            print(f" [Warning] Matplotlib rendering failed for {fname}: {e}")
            
        # 2. Text / ASCII diagram
        ascii_repr = qc.draw(output='text').__str__()
        text_diagrams.append(f"### {title}\n```text\n{ascii_repr}\n```\n")
        
    # Write consolidated ASCII diagrams
    ascii_path = os.path.join(OUTPUT_DIR, "circuit_diagrams_ascii.md")
    with open(ascii_path, "w", encoding="utf-8") as f:
        f.write("# Consolidated Quantum Circuit Text Diagrams\n\n")
        f.write("\n\n".join(text_diagrams))
    print(f"[SUCCESS] All circuit diagrams saved to '{OUTPUT_DIR}'")


# ==============================================================================
# 7. PERSON 3 PARITY ADAPTER
# ==============================================================================

def create_modular_classifier_circuit(
    num_qubits: int,
    features: np.ndarray,
    weights: np.ndarray,
    depth: int = 1,
    entangled: bool = True,
    entangler: str = 'cx',
    topology: str = 'linear'
) -> QuantumCircuit:
    """
    Plug-and-play adapter for Person 3 simulation engine.
    Constructs a fully bound QuantumCircuit ready for execution on AerSimulator.
    
    Args:
        num_qubits: 2 or 4
        features: numpy array of shape (num_qubits,)
        weights: numpy array of shape ((depth + 1) * num_qubits,)
        depth: Ansatz depth L
        entangled: Boolean toggle for entanglement
        entangler: 'cx' or 'cz'
        topology: 'linear', 'circular', or 'full'
    """
    qc = build_full_quantum_circuit(
        num_qubits=num_qubits,
        depth=depth,
        entangled=entangled,
        entangler=entangler,
        topology=topology,
        include_measurement=True
    )
    
    # Extract parameter vectors
    param_dict = {}
    
    # Bind feature parameters
    for i in range(num_qubits):
        param_dict[f"x[{i}]"] = features[i]
        
    # Bind variational weight parameters
    expected_weights = (depth + 1) * num_qubits
    if len(weights) != expected_weights:
        raise ValueError(
            f"Expected {expected_weights} weights for num_qubits={num_qubits}, depth={depth}, "
            f"but received {len(weights)} weights."
        )
        
    for j in range(expected_weights):
        param_dict[f"theta[{j}]"] = weights[j]
        
    # Assign parameters by name
    name_to_param = {p.name: p for p in qc.parameters}
    binding = {name_to_param[k]: v for k, v in param_dict.items() if k in name_to_param}
    
    return qc.assign_parameters(binding)


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("EXECUTING QUANTUM CIRCUIT DESIGN ENGINE")
    print("=" * 70)
    
    # 1. Run full resource audit
    print("\n--- Phase 1: Resource Profiling & Parameter Auditing ---")
    df_audit = run_circuit_resource_audit()
    print("\nSample Audit Rows:")
    print(df_audit.head(8).to_string(index=False))
    
    # 2. Render and export diagrams
    print("\n--- Phase 2: Exporting Circuit Diagrams (PNG & Text) ---")
    export_circuit_diagrams()
    
    # 3. Test Person 3 Parity Adapter
    print("\n--- Phase 3: Testing Parity Adapter with Bound Numerical Values ---")
    dummy_feats_2q = np.array([0.785, 1.571])
    dummy_weights_2q = np.array([0.1, 0.2, 0.3, 0.4]) # 2 qubits, depth 1 -> (1+1)*2 = 4 weights
    qc_bound = create_modular_classifier_circuit(
        num_qubits=2,
        features=dummy_feats_2q,
        weights=dummy_weights_2q,
        depth=1,
        entangled=True,
        entangler='cx'
    )
    print("Bound Circuit for Simulation:")
    print(qc_bound.draw(output='text'))
    
    print("\n" + "=" * 70)
    print("CIRCUIT DESIGN ENGINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 70)
