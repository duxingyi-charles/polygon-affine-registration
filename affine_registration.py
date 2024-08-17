import numpy as np
from scipy.spatial import KDTree
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def centroid(points):
    return np.mean(points, axis=0)

def center_points(points, center):
    return points - center

def compute_diameter(points):
    return np.max(np.linalg.norm(points, axis=1))

def initial_rotation(v_points, u_points):
    pca_v = PCA(n_components=2)
    pca_u = PCA(n_components=2)
    
    pca_v.fit(v_points)
    pca_u.fit(u_points)
    
    R = np.dot(pca_u.components_.T, pca_v.components_)
    return R

def sample_polygon_boundary(vertices, n_points):
    """
    Sample points along the boundary of the polygon defined by vertices.
    
    Args:
        vertices (list or np.ndarray): List of vertices (2D points) defining the polygon.
        n_points (int): Number of points to sample along the boundary.
        
    Returns:
        np.ndarray: Array of sampled points along the polygon boundary.
    """
    vertices = np.array(vertices)
    n_vertices = len(vertices)
    
    # Compute the total perimeter length of the polygon
    perimeter = np.sum(np.linalg.norm(np.roll(vertices, -1, axis=0) - vertices, axis=1))
    
    # Determine the distance between sampled points
    distances = np.linalg.norm(np.roll(vertices, -1, axis=0) - vertices, axis=1)
    cumulative_distances = np.cumsum(distances)
    
    # Distribute the samples along the perimeter
    sampled_points = []
    for i in range(n_vertices):
        start_point = vertices[i]
        end_point = vertices[(i + 1) % n_vertices]
        
        segment_length = distances[i]
        n_segment_points = max(int(np.round(n_points * (segment_length / perimeter))), 1)
        
        for j in range(n_segment_points):
            t = j / n_segment_points
            point = (1 - t) * start_point + t * end_point
            sampled_points.append(point)
    
    return np.array(sampled_points)


def affine_registration(P1, P2, n1, n2, max_iterations=100, tol=1e-6):
    """
    Perform affine registration between two polygons, using boundary sampling.
    
    Args:
        P1 (list): List of vertices for polygon 1.
        P2 (list): List of vertices for polygon 2.
        n1 (int): Number of points to sample on polygon 1.
        n2 (int): Number of points to sample on polygon 2.
        max_iterations (int): Maximum number of iterations for refinement.
        tol (float): Tolerance for convergence.
    
    Returns:
        np.ndarray: Affine transformation matrix.
    """
    # Step 1: Sample points on polygon boundaries
    v_points = sample_polygon_boundary(P1, n1)
    u_points = sample_polygon_boundary(P2, n2)
    #np.savetxt("output/v_points.txt", v_points)
    #np.savetxt("output/u_points.txt", u_points)
    
    # Step 2: Compute centroids and move point sets to the origin
    v_center = centroid(v_points)
    u_center = centroid(u_points)
    
    v_points_centered = center_points(v_points, v_center)
    u_points_centered = center_points(u_points, u_center)
    
    # Step 4: Compute scale factor
    v_diameter = compute_diameter(v_points_centered)
    u_diameter = compute_diameter(u_points_centered)
    scale_factor = u_diameter / v_diameter
    
    # Step 5: Initial rotation using PCA
    R = initial_rotation(v_points_centered, u_points_centered)
    A = scale_factor * R
    
    # Step 6: Iterative Closest Point (ICP)
    tree = KDTree(u_points_centered)
    for iteration in range(max_iterations):
        transformed_v_points = np.dot(v_points_centered, A.T)        
        distances, indices = tree.query(transformed_v_points)
        
        # Update the transformation matrix A
        u_closest = u_points_centered[indices]
        
        # Solve the least squares problem to find the best affine transformation
        A_new = np.linalg.lstsq(v_points_centered, u_closest, rcond=None)[0].T
        
        # Check for convergence
        if np.linalg.norm(A_new - A) < tol:
            A = A_new
            break
        
        A = A_new
    
    # Apply the final transformation (including translation)
    final_transformation = lambda v: np.dot(v - v_center, A.T) + u_center
    translation = u_center - np.dot(v_center, A.T)
    
    return final_transformation, translation, A


def main():
    # Example usage:
    P1 = np.array([[0, 0], [0.5, -0.2], [1, 0], [1, 1], [0, 1]])
    P2 = np.array([[1, 1], [2, 1], [2, 2], [1, 2]])

    #P1 = np.loadtxt("examples/apple.txt")
    #P2 = np.loadtxt("examples/rubin_vase.txt")

    transformation, translation, linear_transform = affine_registration(P1, P2, 100, 100)

    # Transform the points of P1
    P1_transformed = np.dot(np.array(P1), linear_transform.T) + translation
    np.savetxt("output/P1_transformed.txt", P1_transformed)

    # Plot the original polygons and the transformed polygon
    plt.figure(figsize=(8, 8))

    # Plot P1 (original polygon 1)
    plt.plot(np.append(P1[:, 0], P1[0, 0]), np.append(P1[:, 1], P1[0, 1]), 'b-', label='Polygon 1 (Original)')

    # Plot P2 (polygon 2)
    plt.plot(np.append(P2[:, 0], P2[0, 0]), np.append(P2[:, 1], P2[0, 1]), 'k-', label='Polygon 2 (Reference)')

    # Plot transformed P1
    plt.plot(np.append(P1_transformed[:, 0], P1_transformed[0, 0]), np.append(P1_transformed[:, 1], P1_transformed[0, 1]), 'r-', label='Polygon 1 (Transformed)')

    # Set the plot details
    plt.legend()
    plt.title('Polygon Registration Visualization')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.grid(True)
    plt.axis('equal')

    # Show the plot
    plt.show()

if __name__ == "__main__":
    main()