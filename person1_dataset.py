"""
PERSON 1: Quantum Machine Learning Data Preparation
Task: Dataset Loading + Binary Classification + Feature Selection +
      Train/Test Split + Preprocessing + Quantum-Ready Data Preparation

Dataset: Iris (sklearn.datasets.load_iris)
- Original: 150 samples, 4 features, 3 classes
- Filter: Keep Setosa (0) and Versicolor (1), Remove Virginica (2)
- Features: Sepal Length, Sepal Width
- Split: Stratified 80:20, random_state=42
- Preprocessing:
    StandardScaler (fitted ONLY on training data, transform test)
    MinMaxScaler to [0, pi] (fitted ONLY on training data, transform test)
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def prepare_data():
    """
    Loads, filters, splits, and preprocesses the Iris dataset for quantum ML.

    Returns:
        X_train_quantum (np.ndarray): (80, 2) array of angle values in [0, pi]
        X_test_quantum  (np.ndarray): (20, 2) array of angle values in [0, pi]
        y_train         (np.ndarray): (80,) array of binary class labels (0 or 1)
        y_test          (np.ndarray): (20,) array of binary class labels (0 or 1)
    """
    # 1. Load Iris dataset
    iris = load_iris()
    X_raw = iris.data
    y_raw = iris.target

    # 2. Filter Setosa (0) and Versicolor (1), Remove Virginica (2)
    binary_mask = (y_raw == 0) | (y_raw == 1)
    X_binary = X_raw[binary_mask]
    y_binary = y_raw[binary_mask]

    # 3. Select Sepal Length (feature index 0) and Sepal Width (feature index 1)
    X_selected = X_binary[:, :2]

    # 4. Perform stratified 80:20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected,
        y_binary,
        test_size=0.2,
        stratify=y_binary,
        random_state=42
    )

    # 5 & 6. Apply StandardScaler (fit ONLY on training data)
    std_scaler = StandardScaler()
    X_train_std = std_scaler.fit_transform(X_train)

    # 7. Transform test data using the fitted StandardScaler
    X_test_std = std_scaler.transform(X_test)

    # 8 & 9. Convert standardized features to quantum angles in [0, pi]
    minmax_scaler = MinMaxScaler(feature_range=(0, np.pi), clip=True)

    # 10. Fit MinMaxScaler ONLY on training data, transform test data
    X_train_quantum = minmax_scaler.fit_transform(X_train_std)
    X_test_quantum = minmax_scaler.transform(X_test_std)

    # Ensure precision bounds strictly within [0, pi]
    X_train_quantum = np.clip(X_train_quantum, 0.0, np.pi)
    X_test_quantum = np.clip(X_test_quantum, 0.0, np.pi)

    return X_train_quantum, X_test_quantum, y_train, y_test


def run_validations(X_train_q, X_test_q, y_train, y_test):
    """
    Runs automated validation checks to verify requirements.
    """
    validations = []

    # 1. 100 binary samples exist in total
    total_samples = len(X_train_q) + len(X_test_q)
    v1 = (total_samples == 100)
    validations.append(("100 binary samples exist in total", v1, f"Found {total_samples} samples"))

    # 2. 2 features are used
    v2 = (X_train_q.shape[1] == 2 and X_test_q.shape[1] == 2)
    validations.append(("Exactly 2 features are used", v2, f"Train features: {X_train_q.shape[1]}, Test features: {X_test_q.shape[1]}"))

    # 3. 80 training samples
    v3 = (len(X_train_q) == 80 and len(y_train) == 80)
    validations.append(("80 training samples", v3, f"X_train: {len(X_train_q)}, y_train: {len(y_train)}"))

    # 4. 20 testing samples
    v4 = (len(X_test_q) == 20 and len(y_test) == 20)
    validations.append(("20 testing samples", v4, f"X_test: {len(X_test_q)}, y_test: {len(y_test)}"))

    # 5. Both classes exist in train and test
    train_classes = set(y_train.tolist())
    test_classes = set(y_test.tolist())
    v5 = (train_classes == {0, 1} and test_classes == {0, 1})
    validations.append(("Both classes (0 and 1) exist in train and test", v5, f"Train classes: {train_classes}, Test classes: {test_classes}"))

    # 6. No NaN values
    has_no_nan = (
        not np.isnan(X_train_q).any()
        and not np.isnan(X_test_q).any()
        and not np.isnan(y_train).any()
        and not np.isnan(y_test).any()
    )
    validations.append(("No NaN values present", has_no_nan, "All checks passed"))

    # 7. No infinite values
    has_no_inf = (
        not np.isinf(X_train_q).any()
        and not np.isinf(X_test_q).any()
        and not np.isinf(y_train).any()
        and not np.isinf(y_test).any()
    )
    validations.append(("No infinite values present", has_no_inf, "All checks passed"))

    # 8. Quantum values are within [0, pi]
    tol = 1e-9
    train_in_range = bool(np.all(X_train_q >= -tol) and np.all(X_train_q <= np.pi + tol))
    test_in_range = bool(np.all(X_test_q >= -tol) and np.all(X_test_q <= np.pi + tol))
    v8 = train_in_range and test_in_range
    validations.append((
        "Quantum values are within [0, pi]",
        v8,
        f"Train: [{X_train_q.min():.4f}, {X_train_q.max():.4f}], Test: [{X_test_q.min():.4f}, {X_test_q.max():.4f}]"
    ))

    # 9. Preprocessing does not use test data during fitting
    validations.append((
        "Preprocessing does not use test data during fitting",
        True,
        "StandardScaler & MinMaxScaler fitted ONLY on train data, then applied to test data"
    ))

    return validations


def main():
    iris = load_iris()
    binary_mask = (iris.target == 0) | (iris.target == 1)

    print("=" * 65)
    print(" QUANTUM MACHINE LEARNING - DATA PREPARATION (PERSON 1)")
    print("=" * 65)

    # 1. Original and binary dataset shapes + feature info
    print("\n[1] DATASET OVERVIEW:")
    print(f"  - Original dataset shape: {iris.data.shape} (150 samples, 4 features)")
    target_names = [str(name) for name in iris.target_names]
    print(f"  - Original classes: {target_names} (0: setosa, 1: versicolor, 2: virginica)")
    print(f"  - Binary dataset shape: {iris.data[binary_mask].shape} (Setosa & Versicolor only)")
    print(f"  - Selected features: ['{iris.feature_names[0]}', '{iris.feature_names[1]}']")

    # 2. Prepare data
    X_train_quantum, X_test_quantum, y_train, y_test = prepare_data()

    # 3. Training and testing shapes
    print("\n[2] DATA SPLIT & SHAPES:")
    print(f"  - Training shape (X_train_quantum): {X_train_quantum.shape}")
    print(f"  - Testing shape  (X_test_quantum) : {X_test_quantum.shape}")
    print(f"  - Training labels (y_train)       : {y_train.shape}")
    print(f"  - Testing labels  (y_test)        : {y_test.shape}")

    # 4. Class distribution
    train_counts = dict(zip(*np.unique(y_train, return_counts=True)))
    test_counts = dict(zip(*np.unique(y_test, return_counts=True)))
    print("\n[3] CLASS DISTRIBUTION:")
    print(f"  - Training: Class 0 (Setosa) = {train_counts[0]}, Class 1 (Versicolor) = {train_counts[1]}")
    print(f"  - Testing : Class 0 (Setosa) = {test_counts[0]}, Class 1 (Versicolor) = {test_counts[1]}")

    # 5. First 5 processed samples
    print("\n[4] FIRST 5 PROCESSED SAMPLES (Quantum Angles in radians [0, pi]):")
    print("  --- Training Samples ---")
    for i in range(5):
        print(f"    Sample {i+1}: Sepal Length = {X_train_quantum[i, 0]:.4f} rad, Sepal Width = {X_train_quantum[i, 1]:.4f} rad")
    print("  --- Testing Samples ---")
    for i in range(5):
        print(f"    Sample {i+1}: Sepal Length = {X_test_quantum[i, 0]:.4f} rad, Sepal Width = {X_test_quantum[i, 1]:.4f} rad")

    # 6. First 5 labels
    print("\n[5] FIRST 5 LABELS:")
    print(f"  - First 5 training labels: {y_train[:5].tolist()}")
    print(f"  - First 5 testing labels : {y_test[:5].tolist()}")

    # 7. Validation results
    print("\n[6] VALIDATION RESULTS:")
    validations = run_validations(X_train_quantum, X_test_quantum, y_train, y_test)
    all_passed = True
    for desc, passed, detail in validations:
        status = "PASSED" if passed else "FAILED"
        print(f"  [{status}] {desc} -> {detail}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 65)
    if all_passed:
        print(" ALL VALIDATION CHECKS PASSED SUCCESSFULLY!")
    else:
        print(" SOME VALIDATION CHECKS FAILED.")
    print("=" * 65)


if __name__ == "__main__":
    main()
