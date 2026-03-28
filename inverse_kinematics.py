import numpy as np
from scipy.spatial.transform import Rotation as R

def compute_fk(theta1, theta2, theta3, theta4, theta5):
    return np.array([[-np.sin(theta1)*np.sin(theta5)*np.sin(theta2 + theta3 + theta4) + np.cos(theta1)*np.cos(theta5)
                    , -np.sin(theta1)*np.cos(theta5)*np.sin(theta2 + theta3 + theta4) - np.cos(theta1)*np.sin(theta5)
                    , -np.sin(theta1)*np.cos(theta2 + theta3 + theta4)
                    , (0.11257*np.sin(theta2) + 0.0052*np.sin(theta2 + theta3) - 0.028*np.cos(theta2) - 0.1349*np.cos(theta2 + theta3) - 0.1351*np.cos(theta2 + theta3 + theta4) - 0.0306)*np.sin(theta1)]
                    , [np.sin(theta1)*np.cos(theta5) + np.cos(theta1)*np.sin(theta5)*np.sin(theta2 + theta3 + theta4)
                    , -np.sin(theta1)*np.sin(theta5) + np.cos(theta1)*np.cos(theta5)*np.sin(theta2 + theta3 + theta4)
                    , np.cos(theta1)*np.cos(theta2 + theta3 + theta4)
                    , -0.11257*np.sin(theta2)*np.cos(theta1) - 0.0052*np.sin(theta2 + theta3)*np.cos(theta1) + 0.028*np.cos(theta2)*np.cos(theta1) + 0.1349*np.cos(theta2 + theta3)*np.cos(theta1) + 0.1351*np.cos(theta2 + theta3 + theta4)*np.cos(theta1) + 0.0306*np.cos(theta1) + 0.0452]
                    , [-np.sin(theta5)*np.cos(theta2 + theta3 + theta4)
                    , -np.cos(theta5)*np.cos(theta2 + theta3 + theta4)
                    , np.sin(theta2 + theta3 + theta4)
                    , 0.028*np.sin(theta2) + 0.1349*np.sin(theta2 + theta3) + 0.1351*np.sin(theta2 + theta3 + theta4) + 0.11257*np.cos(theta2) + 0.0052*np.cos(theta2 + theta3) + 0.119]
                    , [0, 0, 0, 1]])

def compute_jacobian(theta1, theta2, theta3, theta4, theta5):
    # Precompute terms for rotational rows (Roll, Pitch, Yaw)
    c5 = np.cos(theta5)
    s5 = np.sin(theta5)
    c234 = np.cos(theta2 + theta3 + theta4)
    s234 = np.sin(theta2 + theta3 + theta4)
    
    # Precompute denominators with epsilon to prevent singularity crashes
    denom1_raw = (c234**2) * (s5**2) - 1
    denom1 = denom1_raw if abs(denom1_raw) > 1e-6 else -1e-6
    denom2 = np.sqrt(np.maximum(-(c234**2) * (s5**2) + 1, 1e-6))

    return np.array([
        # Row 1: X position
        [(0.11257*np.sin(theta2) + 0.0052*np.sin(theta2 + theta3) - 0.028*np.cos(theta2) - 0.1349*np.cos(theta2 + theta3) - 0.0601*np.cos(theta2 + theta3 + theta4) - 0.0306)*np.cos(theta1),
         (0.028*np.sin(theta2) + 0.1349*np.sin(theta2 + theta3) + 0.0601*np.sin(theta2 + theta3 + theta4) + 0.11257*np.cos(theta2) + 0.0052*np.cos(theta2 + theta3))*np.sin(theta1),
         (0.1349*np.sin(theta2 + theta3) + 0.0601*np.sin(theta2 + theta3 + theta4) + 0.0052*np.cos(theta2 + theta3))*np.sin(theta1),
         0.0601*np.sin(theta1)*np.sin(theta2 + theta3 + theta4),
         0],
        
        # Row 2: Y position
        [(0.11257*np.sin(theta2) + 0.0052*np.sin(theta2 + theta3) - 0.028*np.cos(theta2) - 0.1349*np.cos(theta2 + theta3) - 0.0601*np.cos(theta2 + theta3 + theta4) - 0.0306)*np.sin(theta1),
         -(0.028*np.sin(theta2) + 0.1349*np.sin(theta2 + theta3) + 0.0601*np.sin(theta2 + theta3 + theta4) + 0.11257*np.cos(theta2) + 0.0052*np.cos(theta2 + theta3))*np.cos(theta1),
         -(0.1349*np.sin(theta2 + theta3) + 0.0601*np.sin(theta2 + theta3 + theta4) + 0.0052*np.cos(theta2 + theta3))*np.cos(theta1),
         -0.0601*np.sin(theta2 + theta3 + theta4)*np.cos(theta1),
         0],
        
        # Row 3: Z position
        [0,
         -0.11257*np.sin(theta2) - 0.0052*np.sin(theta2 + theta3) + 0.028*np.cos(theta2) + 0.1349*np.cos(theta2 + theta3) + 0.0601*np.cos(theta2 + theta3 + theta4),
         -0.0052*np.sin(theta2 + theta3) + 0.1349*np.cos(theta2 + theta3) + 0.0601*np.cos(theta2 + theta3 + theta4),
         0.0601*np.cos(theta2 + theta3 + theta4),
         0],

        # Row 4: Roll orientation (alpha)
        [0,
         -c5 / denom1,
         -c5 / denom1,
         -c5 / denom1,
         -(c234 * s234 * s5) / denom1],

        # Row 5: Pitch orientation (beta)
        [0,
         -(s234 * s5) / denom2,
         -(s234 * s5) / denom2,
         -(s234 * s5) / denom2,
         (c234 * c5) / denom2],

        # Row 6: Yaw orientation (gamma)
        [1,
         -(c234 * c5 * s5) / denom1,
         -(c234 * c5 * s5) / denom1,
         -(c234 * c5 * s5) / denom1,
         -s234 / denom1]
    ])


def calculate_ik_dls(target_pose_6d, initial_joints, max_iter=100, tol=1e-3, damping=0.01):
    theta = np.array(initial_joints, dtype=float)
    
    for i in range(max_iter):
        # 1. Current Forward Kinematics
        T_current = compute_fk(*theta)
        current_pose_6d = matrix_to_pose(T_current)
        
        # 3. Calculate Error
        error = target_pose_6d - current_pose_6d
        
        # 4. Check Stopping Criteria
        if np.linalg.norm(error) < tol:
            print(f"Converged in {i} iterations.")
            break
            
        # 5. Compute Jacobian and DLS Update
        J = compute_jacobian(*theta)
        J_T = J.T
        
        # formula: Δθ = J^T (J J^T + λ^2 I)^-1 e
        lambda_sq_I = (damping**2) * np.eye(6)
        delta_theta = J_T @ np.linalg.inv(J @ J_T + lambda_sq_I) @ error
        
        # 6. Update Joints
        theta += delta_theta
        
    return theta

def matrix_to_pose(T):

    pos_current = T[:3, 3]
    rot_matrix = T[:3, :3]
        
    # Note: Ensure the 'xyz' order matches the roll/pitch/yaw definition in your Jacobian
    euler_current = R.from_matrix(rot_matrix).as_euler('xyz', degrees=False)
    current_pose_6d = np.concatenate((pos_current, euler_current))

    return current_pose_6d

def calculate_ik_dls_withlimits(target_pose_6d, initial_joints, joint_limits, max_iter=500, tol=1e-3, damping=0.01):
    theta = np.array(initial_joints, dtype=float)
    limits = np.array(joint_limits)
    
    for i in range(max_iter):
        # 1. Current Forward Kinematics
        T_current = compute_fk(*theta)
        
        # 2. Extract 6D Pose
        pos_current = T_current[:3, 3]
        rot_matrix = T_current[:3, :3]
        euler_current = R.from_matrix(rot_matrix).as_euler('xyz', degrees=False)
        current_pose_6d = np.concatenate((pos_current, euler_current))
        
        # 3. Calculate Error
        error = target_pose_6d - current_pose_6d
        
        # 4. Check Stopping Criteria
        if np.linalg.norm(error) < tol:
            break
            
        # 5. Compute Jacobian and DLS Update
        J = compute_jacobian(*theta)
        J_T = J.T
        lambda_sq_I = (damping**2) * np.eye(6)
        delta_theta = J_T @ np.linalg.inv(J @ J_T + lambda_sq_I) @ error
        
        # 6. Update Joints
        theta += delta_theta
        
        # 7. Clamp to Joint Limits
        theta = np.clip(theta, limits[:, 0], limits[:, 1])
        
    return theta

joint_limits = np.array([
    [-2.0, 2.0],       # theta1: Shoulder rotation (SR)
    [-1.570, 1.570],   # theta2: Shoulder pitch (SP)
    [-1.580, 1.580],   # theta3: Elbow (E)
    [-1.570, 1.570],   # theta4: Wrist pitch (WP)
    [-3.142, 3.142]    # theta5: Wrist roll (WR)
])

target_poses = [
    np.array([0.2, 0.2, 0.2, 0.000, 1.570, 0.650]),    # Pose I
    np.array([0.2, 0.1, 0.4, 0.000, 0.000, -1.570]),   # Pose II
    np.array([0.0, 0.0, 0.4, 0.000, -0.785, 1.570]),   # Pose III
    np.array([0.0, 0.0, 0.07, 3.141, 0.000, 0.000]),   # Pose IV
    np.array([0.0, 0.0452, 0.45, -0.785, 0.000, 3.141]) # Pose V    
]

initial_guess = [0 , 0 , 0 , 0 , 0]
np.set_printoptions(suppress=True, precision=5)
for pose in target_poses:

    thetas_nolimits = calculate_ik_dls(pose, initial_guess)
    thetas = calculate_ik_dls_withlimits(pose , initial_guess , joint_limits)
    print("Wanted Pose = " , pose)
    print("Achieved Pose(with limits) = " , matrix_to_pose(compute_fk(*thetas_nolimits)))
    #print("Achieved Pose(without limits) = " , matrix_to_pose(compute_fk(*thetas_nolimits)))
    #print("joint positions (no limits) = " , thetas_nolimits)
    print("joint positions (with limits) =" , thetas)
    print("\n")