import warp as wp
import numpy as np

def manifold_points(poly: np.ndarray, poly_mask: np.ndarray, poly_count: int, poly_norm: np.ndarray) -> np.ndarray:
    """Chooses four points on the polygon with approximately maximal area."""
    dist_mask = np.where(poly_mask, 0.0, -1e6)
    
    a_idx = np.argmax(dist_mask)
    a = poly[a_idx]
    
    # Choose point b furthest from a
    b_idx = (((a - poly) ** 2).sum(axis=1) + dist_mask).argmax()
    b = poly[b_idx]
    
    # Choose point c furthest along the axis orthogonal to (a-b)
    ab = np.cross(poly_norm, a - b)
    ap = a - poly
    c_idx = (np.abs(np.dot(ap, ab)) + dist_mask).argmax()
    c = poly[c_idx]
    
    # Choose point d furthest from the other two triangle edges
    ac = np.cross(poly_norm, a - c)
    bc = np.cross(poly_norm, b - c)
    bp = b - poly
    dist_bp = np.abs(np.dot(bp, bc)) + dist_mask
    dist_ap = np.abs(np.dot(ap, ac)) + dist_mask
    d_idx = (dist_bp + dist_ap).argmax() % poly_count
    
    return np.array([a_idx, b_idx, c_idx, d_idx])


def sel(condition: bool, onTrue: float, onFalse: float) -> float:
    """Returns onTrue if condition is true, otherwise returns onFalse."""
    return onTrue if condition else onFalse


def manifold_points3(poly: np.ndarray, poly_mask: np.ndarray, poly_count: int, poly_norm: np.ndarray) -> np.ndarray:
    """Chooses four points on the polygon with approximately maximal area."""
    max_val = -1e6
    a_idx = 0
    for i in range(poly_count):
        val = sel(poly_mask[i] > 0.0, 0.0, -1e6)
        if val > max_val:
            max_val = val
            a_idx = i
    a = poly[a_idx]

    # choose point b furthest from a
    max_val = -1e6
    b_idx = 0
    for i in range(poly_count):
        val = np.sum((a - poly[i]) * (a - poly[i])) + sel(poly_mask[i] > 0.0, 0.0, -1e6)
        if val > max_val:
            max_val = val
            b_idx = i
    b = poly[b_idx]

    # choose point c furthest along the axis orthogonal to (a-b)
    ab = np.cross(poly_norm, a - b)
    ap = a - poly
    max_val = -1e6
    c_idx = 0
    for i in range(poly_count):
        val = np.abs(np.dot(ap[i], ab)) + sel(poly_mask[i] > 0.0, 0.0, -1e6)
        if val > max_val:
            max_val = val
            c_idx = i
    c = poly[c_idx]

    # choose point d furthest from the other two triangle edges
    ac = np.cross(poly_norm, a - c)
    bc = np.cross(poly_norm, b - c)
    bp = b - poly
    max_val = -1e6
    d_idx = 0
    for i in range(poly_count):
        val = np.abs(np.dot(bp[i], bc)) + np.abs(np.dot(ap[i], ac)) + sel(poly_mask[i] > 0.0, 0.0, -1e6)
        if val > max_val:
            max_val = val
            d_idx = i
    return np.array([a_idx, b_idx, c_idx, d_idx])



@wp.kernel
def matrix_vector_multiply(mat: wp.mat33, vec: wp.vec3):
    """Multiplies a 3x3 matrix with a 3D vector using warp's matrix operations."""
    result = mat @ vec
    print(result)

# Test the kernel
wp.init()

# Create test data
matrix = wp.mat33(1.0, 0.0, 0.0,
                    0.0, 11.0, 1.0, 
                    0.0, 0.0, 1.0)
vector = wp.vec3(1.0, 2.0, 3.0)

# Launch kernel
wp.launch(
    kernel=matrix_vector_multiply,
    dim=1,
    inputs=[matrix, vector],
    device='cpu'
)

print(f"Input matrix:\n{matrix}")
print(f"Input vector: {vector}")






# Example input data
N = 5  # Number of points
M = 3  # 3D space

# Define polygon points (N, M) - random example
poly = np.array([[1.0, 2.0, 3.0],
                 [4.0, 5.0, 6.0],
                 [7.0, 8.0, 9.0],
                 [10.0, 11.0, 12.0],
                 [13.0, 14.0, 15.0]], dtype=np.float32)

# Define boolean mask (N,)
poly_mask = np.array([True, False, True, False, True], dtype=bool)

# Define normal vector (M,)
poly_norm = np.array([0.0, 0.0, 1.0], dtype=np.float32)

# Call the function
selected_indices = manifold_points(poly, poly_mask, poly.shape[0], poly_norm)
selected_indices2 = manifold_points3(poly, poly_mask, poly.shape[0], poly_norm)

# Print results
print("Selected point indices:", selected_indices)
print("Selected point indices2:", selected_indices2)
print("Selected points:", poly[selected_indices])