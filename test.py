import numpy as np

def cosine_similarity(A, B):
    dot_product = np.dot(A, B)
    norm_A = np.linalg.norm(A)
    norm_B = np.linalg.norm(B)
    return dot_product / (norm_A * norm_B)

def rff_projection(A, dimensions=10, sigma=1.0):
    W = np.random.normal(0, 1/sigma, size=(dimensions, len(A)))
    b = np.random.uniform(0, 2 * np.pi, dimensions)
    projection = np.sqrt(2 / dimensions) * np.cos(np.dot(W, A) + b)
    return projection
  
A = np.array([1, 0])
B = np.array([0.9, 0.1])
C = np.array([0, 1])

cos_A_B_2D = cosine_similarity(A, B)
cos_A_C_2D = cosine_similarity(A, C)

print("Cosine similarity 2D:")
print(f"A-B: {cos_A_B_2D:.3f}")
print(f"A-C: {cos_A_C_2D:.3f}")

A_high = rff_projection(A, 10)
B_high = rff_projection(B, 10)
C_high = rff_projection(C, 10)

cos_A_B_10D = cosine_similarity(A_high, B_high)
cos_A_C_10D = cosine_similarity(A_high, C_high)

print("\nCosine similarity in 10D:")
print(f"A-B: {cos_A_B_10D:.3f}")
print(f"A-C: {cos_A_C_10D:.3f}")
