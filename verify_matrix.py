# verify_matrix.py
import numpy as np

# Step 1: Load the computed matrix
m = np.load("dissimilarity_matrix.npy")

# Step 2: Basic info
print("Matrix shape:", m.shape)
print("Min value:", np.min(m))
print("Max value:", np.max(m))
print("Mean value:", np.mean(m))

# Step 3: Check symmetry
is_symmetric = np.allclose(m, m.T, atol=1e-6)
print("Symmetric:", is_symmetric)

# Step 4: Check diagonal zeros
diag_zero = np.allclose(np.diag(m), 0.0, atol=1e-6)
print("Diagonal zeros:", diag_zero)

# Optional: print small preview
print("\nTop-left corner of matrix (5x5):")
print(m[:5, :5])
