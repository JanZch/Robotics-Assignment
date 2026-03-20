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

def angle_index(expr, thetas):
    """
    Convert angle expression like:
        theta2 + theta3 + theta4
    into string:
        '234'
    """
    expr = simplify(expr)

    if expr in thetas:
        return str(thetas.index(expr) + 1)

    if isinstance(expr, Add):
        indices = []
        for term in expr.args:
            if term in thetas:
                indices.append(str(thetas.index(term) + 1))
        return "".join(indices)

    raise ValueError(f"Unsupported trig argument: {expr}")


def compress_trig(expr, thetas):
    """
    Replace sin/cos(angle sums) with compact symbols.
    """
    replacements = {}

    for trig in expr.atoms(sin, cos):

        arg = trig.args[0]
        idx = angle_index(arg, thetas)

        if trig.func == sin:
            new_sym = Symbol(f"s_{idx}")
        else:
            new_sym = Symbol(f"c_{idx}")

        replacements[trig] = new_sym

    return expr.xreplace(replacements)

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

    # Extract rotation matrix
    R = T_total[0:3, 0:3]

    # Compute Euler angles (convention: roll applied first, then pitch, then yaw)
    roll = atan2(R[2, 1], R[2, 2])
    pitch = asin(-R[2, 0])
    yaw = atan2(R[1, 0], R[0, 0])

    # Simplify the pose components
    pose = [simplify(x), simplify(y), simplify(z), simplify(roll), simplify(pitch), simplify(yaw)]

    return pose

def compute_jacobian(pose_expressions, joint_angles):
    """
    Compute the Jacobian matrix by taking partial derivatives.
    
    J[i,j] = ∂pose[i]/∂θ[j]
    
    Returns a 6x5 Jacobian matrix where:
    - Rows: [x, y, z, roll, pitch, yaw]
    - Columns: [θ1, θ2, θ3, θ4, θ5]
    """
    jacobian_matrix = Matrix.zeros(6, 5)
    
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

def generate_latex_jacobian(jacobian_symbolic):
    """
    Generate LaTeX representation of the Jacobian matrix.
    
    Returns a string containing LaTeX code that can be directly
    pasted into a LaTeX document using equation environment.
    """
    latex_str = latex(jacobian_symbolic, mode='equation*')
    return latex_str

def generate_individual_latex_jacobians(jacobian_symbolic):
    """
    Generate individual LaTeX expressions for each cell of the Jacobian matrix.
    
    Returns a dictionary with labels as keys and LaTeX expressions as values.
    Format: J_pose_theta = expression
    """
    labels_pose = ['x', 'y', 'z', 'roll', 'pitch', 'yaw']
    labels_theta = ['\\theta_1', '\\theta_2', '\\theta_3', '\\theta_4', '\\theta_5']
    
    jacobian_cells = {}
    
    for i in range(6):
        for j in range(5):
            key = "J_{"+ labels_pose[i] + labels_theta[j] + "}"
            element = jacobian_symbolic[i, j]
            latex_element = latex(element)
            jacobian_cells[key] = latex_element
    
    return jacobian_cells

# Example usage: get symbolic expressions
print("Computing symbolic pose expressions...")
pose_expressions = symbolic_fwd_kinematics(thetas)
print("\nSymbolic pose expressions (6 components):")
for i, expr in enumerate(pose_expressions):
    labels = ['x', 'y', 'z', 'roll', 'pitch', 'yaw']
    print(f"{labels[i]}")

print("\nComputing Jacobian matrix (this may take a moment)...")
jacobian = compute_jacobian(pose_expressions, thetas)
print("Jacobian Matrix dimensions: 6 rows (pose components) × 5 columns (joint angles)")

print("\nSimplifying Jacobian matrix...")
jacobian_simplified = simplify(jacobian)
print("\nCompressing trigonometric expressions...")

jacobian_compact = jacobian_simplified.applyfunc(
    lambda e: compress_trig(e, thetas)
)

# Evaluate at zero position
print("\nEvaluating Jacobian at zero joint angles...")
zero_angles = [0.0, 0.0, 0.0, 0.0, 0.0]
jacobian_zero = evaluate_jacobian_numeric(jacobian_simplified, thetas, zero_angles)
print("\nJacobian at zero position (6×5):")
print(jacobian_zero)

print("\n" + "="*80)
print("SIMPLIFIED SYMBOLIC JACOBIAN MATRIX")
print("="*80)
print("\nRows: [x, y, z, roll, pitch, yaw]")
print("Columns: [θ1, θ2, θ3, θ4, θ5]\n")
pprint(jacobian_compact)

# Print column-wise with labels for clarity
print("\n" + "="*80)
print("JACOBIAN COMPONENTS (Individual Elements)")
print("="*80)
labels_pose = ['x', 'y', 'z', 'roll', 'pitch', 'yaw']
labels_theta = ['θ1 (SR)', 'θ2 (SP)', 'θ3 (E)', 'θ4 (WP)', 'θ5 (WR)']

for i in range(6):
    print(f"\nRow {i} - {labels_pose[i]} derivatives:")
    for j in range(5):
        element = jacobian_simplified[i, j]
        print(f"  ∂{labels_pose[i]}/∂{labels_theta[j]} = {element}")

# ============================================================================
# LATEX OUTPUT - Ready to paste into LaTeX document
# ============================================================================
print("\n\n")
print("="*80)
print("LaTeX OUTPUT - Individual Jacobian Elements")
print("="*80)

jacobian_cells = generate_individual_latex_jacobians(jacobian_compact)

# Print to console
print("\nIndividual LaTeX Expressions:\n")
for key, latex_expr in jacobian_cells.items():
    print(f"{key} = {latex_expr}\n")

# Save individual expressions to file
with open('jacobian_matrix.tex', 'w') as f:
    f.write("% Simplified Symbolic Jacobian Matrix for 5-DOF RRRRR Robotic Arm\n")
    f.write("% Generated from symbolic kinematics computation\n")
    f.write("% \n")
    f.write("% Pose vector: x = [x, y, z, roll, pitch, yaw]\n")
    f.write("% Joint angles: q = [θ1, θ2, θ3, θ4, θ5]\n")
    f.write("% \n")
    f.write("% Jacobian J = ∂x/∂q (6×5 matrix)\n")
    f.write("% Individual matrix elements:\n\n")
    
    for key, latex_expr in jacobian_cells.items():
        f.write(f"{key} &= {latex_expr}\\\ \n")

print("\n" + "-"*80)
print("LaTeX output saved to: jacobian_matrix.tex")
print("-"*80)

# print("Numerical pose at zero position:", [float(p) for p in numerical_pose])

