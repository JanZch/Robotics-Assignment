import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize
import itertools
import csv

#This script calculates the IK for the 5 poses using damped least squares IK
#The solutions in order can be seen in the output.csv and are also printed to the cmd line
#at the end of the file there is an attempt to sweep the initial guess given to the IK solver
#as an attempt to find multiple solutions, which fails and takes a long time to compute.

def compute_fk(theta1, theta2, theta3, theta4, theta5):
    '''
    returns the homogenous matrix for the full forward kinematics. To use
    '''
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


def matrix_to_pose(T):

    pos_current = T[:3, 3]
    rot_matrix = T[:3, :3]
        
    # Note: Ensure the 'xyz' order matches the roll/pitch/yaw definition in your Jacobian
    euler_current = R.from_matrix(rot_matrix).as_euler('xyz', degrees=False)
    current_pose_6d = np.concatenate((pos_current, euler_current))

    return current_pose_6d

def calculate_ik_dls_withlimits(target_pose_6d, initial_joints, joint_limits, max_iter=100, tol=1e-3, damping=0.01):
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
        
    return theta, np.linalg.norm(error)


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

guesses = [
    np.array([-0.920,  0.645, -0.876,  0.232,  1.571]),  # Pose 1
    np.array([-1.303, -0.173,  0.430,  1.314, -0.267]),  # Pose 2
    np.array([-0.001,  0.953, -0.185,  1.570,  1.571]),  # Pose 3
    np.array([ 3.142,  1.506, -1.158, -1.918, -3.142]),  # Pose 4
    np.array([ 0.000,  0.227,  0.821,  1.308,  3.142])   # Pose 5
]


print("\n")
initial_guess = [0 , 0 , 0 , 0 , 0]
solutions_for_poses = []
np.set_printoptions(suppress=True, precision=5)
for index , pose in enumerate(target_poses):
    thetas, error = calculate_ik_dls_withlimits(pose, guesses[index] , joint_limits, max_iter=1000)
    print(thetas)
    solutions_for_poses.append(thetas)
    print("Wanted Pose = " , pose)
    print("Achieved Pose(with limits) = " , matrix_to_pose(compute_fk(*thetas)))
    print("\n")
print(solutions_for_poses)
np.savetxt("output.csv", thetas, delimiter=",")

with open('output.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(solutions_for_poses)

'''
#Systematically find solutions
pose_tags = ['Pose I', 'Pose II', 'Pose III', 'Pose IV', 'Pose V']


def find_ik_solutions_systematic(ik_function, poses, limits, tags, steps_per_joint=4, tol=1e-3):
    all_solutions = []
    
    joint_spaces = [np.linspace(lim[0], lim[1], steps_per_joint) for lim in limits]
    initial_guesses = list(itertools.product(*joint_spaces))
    total_guesses = len(initial_guesses)
    
    for pose, tag in zip(poses, tags):
        unique_solutions_for_pose = []
        
        for i, guess in enumerate(initial_guesses):
            print(i)
            # Built-in console progress (updates on the same line)
            if i % 100 == 0 or i == total_guesses - 1:
                print(f"\r{tag}: Testing condition {i + 1}/{total_guesses}", end="", flush=True)
                
            sol, error = ik_function(pose, np.array(guess), limits)
            if error < 1e-2:
                
                is_duplicate = any(np.allclose(sol, existing[1], atol=tol) for existing in unique_solutions_for_pose)
                    
                if not is_duplicate:
                    unique_solutions_for_pose.append((tag, sol))
                        
        all_solutions.extend(unique_solutions_for_pose)
        # Newline and summary when done with the current pose
        print(f"\n-> Found {len(unique_solutions_for_pose)} distinct solutions.\n")
        
    return all_solutions


print("START SEARCH")
solutions = find_ik_solutions_systematic(calculate_ik_dls_withlimits, target_poses, joint_limits, pose_tags)

with open('solutions.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(solutions)
'''