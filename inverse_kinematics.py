import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize

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


def calculate_ik_dls(target_pose_6d, initial_joints, max_iter=20, tol=1e-3, damping=0.01):
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

def calculate_ik_dls_withlimits(target_pose_6d, initial_joints, joint_limits, max_iter=5000, tol=1e-3, damping=0.01):
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
        #error = target_pose_6d[:3] - pos_current
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
        
    return theta, error

def ik_solver(target_position, q_initial_guess=np.zeros(5)):
    """
    Solves IK numerically by minimizing the positional error.
    """
    def objective_function(q):
        current_pos = matrix_to_pose(compute_fk(*q))
        # Calculate Euclidean distance between current and target position
        error = np.linalg.norm(current_pos - np.array(target_position))
        return error

    # Replace with your actual joint limits measured in Task 1.1.1
    joint_bounds = [(-np.pi, np.pi)] * 5 

    # Run the optimization (SLSQP handles bounds well)
    result = minimize(
        objective_function, 
        q_initial_guess, 
        method='SLSQP', 
        bounds=joint_bounds,
        tol=1e-4
    )

    # Check if a valid solution was found within an acceptable error margin
    if result.success and result.fun < 1e-3:
        return result.x  # Returns the array of 5 joint angles
    else:
        print("DANGER")
        return None      # Position is likely unreachable

joint_limits = np.array([
    [-2.0, 2.0],       # theta1: Shoulder rotation (SR)
    [-1.570, 1.570],   # theta2: Shoulder pitch (SP)
    [-1.580, 1.580],   # theta3: Elbow (E)
    [-1.570, 1.570],   # theta4: Wrist pitch (WP)
    [-3.142, 3.142]    # theta5: Wrist roll (WR)
])

target_poses = [
    #np.array([0.2, 0.2, 0.2, 0.000, 1.570, 0.650]),    # Pose I
    np.array([0 , 0.2698756  , 0.37727276, 0.000, 0, 0]),
    np.array([0  ,  0.23711848  ,  0.29340536, 0.000, 0, 0])
    #np.array([0.2, 0.1, 0.4, 0.000, 0.000, -1.570]),   # Pose II
    #np.array([0.0, 0.0, 0.4, 0.000, -0.785, 1.570]),   # Pose III
    #np.array([0.0, 0.0, 0.07, 3.141, 0.000, 0.000]),   # Pose IV
    #np.array([0.0, 0.0452, 0.45, -0.785, 0.000, 3.141]) # Pose V
    #np.array([0.2 , 0.15 , 0.2 ])    
]

#start position square:  EE =[0 , 0.2698756  , 0.37727276 , -1.41579633 , 0 , 0 ] , joint state: [0 ,  0.23 ,  0.85 , -0.925 , 0]
# another possible position: EE = [ 0  ,  0.23711848  ,  0.29340536 , -1.93279633 , 0  ,  0  ] ; joint state: [0 , 0.586 , 0.367 , -1.315 , 0]
#another position to the right: EE = [ 0.16835098 ,  0.13734471 ,  0.29340536 , -2.22134663 , -0.94621135 , -0.51689397] ; joint state: [-1.07 , 0.586 , 0.367 , -1.315 , -1.05] 
print("Start Position = " , matrix_to_pose(compute_fk( 0 , 0.23 , 0.85 , -0.925 , 0)))    
print("Another Position = " , matrix_to_pose(compute_fk(0 , 0.586 , 0.367 , -1.315 , 0)))   
print("another position , to the right of the robots" , matrix_to_pose(compute_fk(-1.07 , 0.586 , 0.367 , -1.315 , -1.05)) )
print("HOME = " , matrix_to_pose(compute_fk(0 , 0 , 0 , 0 , 0)))
print("To see 0,0,0 orientation= " , matrix_to_pose(compute_fk(0 , 0 , 0 , 1.57 , 0)))

print("\n")
initial_guess = [0 , 0 , 0 , 0 , 0]

np.set_printoptions(suppress=True, precision=5)
for pose in target_poses:
    thetas_nolimits,error = calculate_ik_dls_withlimits(pose, initial_guess , joint_limits)
    print(thetas_nolimits)
    thetas = calculate_ik_dls_withlimits(pose , initial_guess , joint_limits)
    print("Wanted Pose = " , pose)
    print("Achieved Pose(with limits) = " , matrix_to_pose(compute_fk(*thetas_nolimits)))
    #print("Achieved Pose(without limits) = " , matrix_to_pose(compute_fk(*thetas_nolimits)))
    #print("joint positions (no limits) = " , thetas_nolimits)
    print("joint positions (with limits) =" , thetas)
    print("\n")


def get_square_points(center, side_length, num_points=20):
    cx, cy, cz = center
    h = side_length / 2.0
    
    # Define corners: Top-Right -> Top-Left -> Bottom-Left -> Bottom-Right -> Top-Right
    corners = [
        (cx + h, cy + h),
        (cx - h, cy + h),
        (cx - h, cy - h),
        (cx + h, cy - h),
        (cx + h, cy + h)
    ]
    
    pts_per_side = num_points // 4
    x_vals, y_vals = [], []
    
    for i in range(4):
        x_start, y_start = corners[i]
        x_end, y_end = corners[i+1]
        
        # endpoint=False prevents overlapping corner points
        x_vals.extend(np.linspace(x_start, x_end, pts_per_side, endpoint=False))
        y_vals.extend(np.linspace(y_start, y_end, pts_per_side, endpoint=False))
        
    z_vals = np.full(num_points, cz)
    
    return np.column_stack((x_vals, y_vals, z_vals))

# Generate the trajectory
center = [0.0, 0.23, 0]
side_length = 0.1  # 10 cm
points = get_square_points(center, side_length)

print(points.shape) # Output: (100, 3)
print(points[:5])   # Preview first 5 points along the top edge

radius = 0.1
c_x= 0.2
c_y = 0.15
c_z = 0.2 

circle_points = get_square_points([c_x , c_y ,c_z] , radius , 20)
print("START Square \n")

initial_guess = [0 ,  0.23 ,  0.85 , -0.925 , 0]
theta_joints = []
for index , point in enumerate(circle_points):
    
    #if(index ==1):
     #   break
    pose = np.array([point[0], point[1] , point[2] , -np.pi ,0 ,0])
    thetas = ik_solver(pose , initial_guess) #add joint limits and 
    error = np.linalg.norm(error)
    if index == 0:
        if error <1e-2:
            theta_joints.append(thetas)
    else:
        if error < 1e-2 and  (thetas-theta_joints[len(theta_joints)-1]) < 0.1 :
            theta_joints.append(thetas)
            print(index , ": Close Enough, error = " ,error)
        else:
            theta_joints.append(thetas)
            print(index ,": Large difference, error = " , error)
    print("Wanted Pose = " , pose)
    print("Achieved Pose(with limits) = " , matrix_to_pose(compute_fk(*thetas)))
    print("joint positions (with limits) =" , thetas)
    print("\n")
    initial_guess = thetas

#save data
np.savetxt("output.csv", theta_joints, delimiter=",")
    



# Pick square
# Generate points on square trajectory
# Do IK on one of them, make sure it works.
# Pick next closest point and use the last IK solution as initial guess
# Make sure the position kinematics is solved, If possible make sure the orientation does not vary too much or stays constant.
# Once the IK is solved for all these points, implement the series of points as a script in simulation. 