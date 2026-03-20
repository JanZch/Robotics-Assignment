import numpy as np
from sympy import symbols, cos, sin, Matrix, simplify, atan2, sqrt, pi, pprint, asin, latex

# Static rotation matrices (numerical, but we'll use them in sympy context)
R_B_W_static = Matrix([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
R_S_B_static = Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
R_UA_S_static = Matrix([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
R_LA_UA_static = Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
R_WR_LA_static = Matrix([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
R_G_WR_static = Matrix([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
R_GC_G_static = Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

# Translation vectors
P_BORG_W = Matrix([[0], [0], [0]])
P_SORG_B = Matrix([[0], [-0.0452], [0.0165]])
P_UAORG_S = Matrix([[0], [-0.0306], [0.1025]])
P_LAORG_UA = Matrix([[0.11257], [-0.028], [0]])
P_WRORG_LA = Matrix([[0.0052], [-0.1349], [0]])
P_GORG_WR = Matrix([[-0.0601], [0], [0]])
P_GCORG_G = Matrix([[0], [0], [0.075]])

def get_homogenous_transform(R, P):
    """Create homogeneous transformation matrix"""
    return Matrix([
        [R[0,0], R[0,1], R[0,2], P[0]],
        [R[1,0], R[1,1], R[1,2], P[1]],
        [R[2,0], R[2,1], R[2,2], P[2]],
        [0, 0, 0, 1]
    ])

# Static homogeneous transforms
T_B_W_static = get_homogenous_transform(R_B_W_static, P_BORG_W)
T_S_B_static = get_homogenous_transform(R_S_B_static, P_SORG_B)
T_UA_S_static = get_homogenous_transform(R_UA_S_static, P_UAORG_S)
T_LA_UA_static = get_homogenous_transform(R_LA_UA_static, P_LAORG_UA)
T_WR_LA_static = get_homogenous_transform(R_WR_LA_static, P_WRORG_LA)
T_G_WR_static = get_homogenous_transform(R_G_WR_static, P_GORG_WR)
T_GC_G_static = get_homogenous_transform(R_GC_G_static, P_GCORG_G)

static_transforms = [T_B_W_static, T_S_B_static, T_UA_S_static, T_LA_UA_static, T_WR_LA_static, T_G_WR_static, T_GC_G_static]

# Joint angles (5 DoF: SR, SP, E, WP, WR)
theta1, theta2, theta3, theta4, theta5 = symbols('theta1 theta2 theta3 theta4 theta5')
thetas = [theta1, theta2, theta3, theta4, theta5]

from sympy import Add, Symbol

def symbolic_fwd_kinematics(thetas):
    """
    Compute symbolic forward kinematics.
    Returns the pose vector x = [x, y, z, roll, pitch, yaw]
    where x = f(q), q = [theta1, theta2, theta3, theta4, theta5]
    """
    T_links = [T_B_W_static]

    for i, theta in enumerate(thetas):
        # Joint rotation around Z-axis
        R_joint = Matrix([
            [cos(theta), -sin(theta), 0],
            [sin(theta), cos(theta), 0],
            [0, 0, 1]
        ])
        T_joint = get_homogenous_transform(R_joint, Matrix([[0], [0], [0]]))

        # Combine static transform with joint rotation
        T_link = static_transforms[i+1] * T_joint
        T_links.append(T_link)

    # Compute total transform from world to gripper center
    T_total = Matrix.eye(4)
    for T in T_links:
        T_total = T_total * T

    # Simplify the total transformation matrix
    T_total = simplify(T_total)

    # Extract position
    x = T_total[0, 3]
    y = T_total[1, 3]
    z = T_total[2, 3]

    # Simplify the pose components
    pose = [simplify(x), simplify(y), simplify(z)]

    return pose

def compute_jacobian(pose_expressions, joint_angles):
    """
    Compute the Jacobian matrix by taking partial derivatives.
    
    J[i,j] = ∂pose[i]/∂θ[j]
    
    Returns a 3x5 Jacobian matrix where:
    - Rows: [x, y, z]
    - Columns: [θ1, θ2, θ3, θ4, θ5]
    """
    jacobian_matrix = Matrix.zeros(3, 5)
    
    for i, pose_component in enumerate(pose_expressions):
        for j, theta in enumerate(joint_angles):
            # Compute partial derivative and simplify
            partial_derivative = simplify(pose_component.diff(theta))
            jacobian_matrix[i, j] = partial_derivative
    
    return jacobian_matrix

def evaluate_jacobian_numeric(jacobian_symbolic, joint_angles, angle_values):
    """
    Evaluate Jacobian at specific joint angles.
    
    Args:
        jacobian_symbolic: Symbolic Jacobian matrix
        joint_angles: List of symbolic angle variables
        angle_values: List of numerical values for angles
    
    Returns:
        Numerical Jacobian matrix as numpy array
    """
    subs_dict = {joint_angles[i]: angle_values[i] for i in range(len(joint_angles))}
    jacobian_numeric = jacobian_symbolic.subs(subs_dict)
    return np.array(jacobian_numeric.tolist(), dtype=float)


# Example usage: get symbolic expressions
print("Computing symbolic pose expressions...")
pose_expressions = symbolic_fwd_kinematics(thetas)
print("\nSymbolic pose expressions (6 components):")
for i, expr in enumerate(pose_expressions):
    labels = ['x', 'y', 'z', 'roll', 'pitch', 'yaw']
    print(f"{labels[i]}")

print("\nComputing Jacobian matrix (this may take a moment)...")
jacobian = compute_jacobian(pose_expressions, thetas)
print("Jacobian Matrix dimensions: 3 rows (pose components) × 5 columns (joint angles)")

print("\nSimplifying Jacobian matrix...")
jacobian_simplified = simplify(jacobian)
print("\nCompressing trigonometric expressions...")



