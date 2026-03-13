import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, cos, sin, Matrix, simplify
from sympy import pprint

symbols
#Rotation Matrices calculated on paper.
#These are the rotation matrices from one joint to another if all joints are zeroed.
R_B_W_static = np.array([[-1 , 0 , 0],[0, -1, 0],[0 , 0 , 1]])  
R_S_B_static = np.array([[1 , 0 , 0] , [0 , 1 , 0] , [0 , 0 , 1]])
R_UA_S_static = np.array([[0, 0, -1] , [0 , 1 , 0] , [1 , 0 , 0]])
R_LA_UA_static = np.array([[1, 0, 0] , [0, 1, 0] , [0, 0, 1]])
R_WR_LA_static = np.array([[0 , -1, 0] , [1 , 0 , 0] , [0 , 0 , 1]])
R_G_WR_static = np.array([[0, 0, -1] , [0 , 1 , 0] , [1 , 0 , 0]])
R_GC_G_static = np.array([ [1, 0 ,0], [0 , 1, 0] , [ 0 , 0, 1]])
R_J_G_static = np.array([[-1 , 0 , 0] , [ 0 , 0 , -1] , [0 , -1 , 0]])

 
#Vectors pointing from one system of reference to the next
P_BORG_W = np.array([[0] , [0] , [0]])
P_SORG_B = np.array([[0] , [-0.0452] ,  [0.0165]])
P_UAORG_S =np.array([[0],  [-0.0306] , [0.1025]])
P_LAORG_UA = np.array([[0.11257] , [-0.028 ] , [0]])
P_WRORG_LA = np.array([[0.0052] , [-0.1349] , [0]])
P_GORG_WR = np.array([[-0.0601], [0] , [0]])
P_GCORG_G =np.array([[0] ,[0] , [0.075]])
P_JORG_G = np.array([[-0.0202] , [0] , [0.0244]])

# Measured Joint bounds in radians [min, max]
SR_bounds = np.array([-1.97, 2.14])  # Shoulder rotation
SP_bounds = np.array([-2.05, 1.8])   # Shoulder pitch
E_bounds  = np.array([-1.75, 1.63])  # Elbow
WP_bounds = np.array([-1.77, 1.8])   # Wrist pitch
WR_bounds = np.array([-2.86, 2.96])  # Wrist roll
G_bounds  = np.array([0.0, 2.200])   # Gripper

# Simulation Joint bounds in radians [min, max]
SR_bounds = np.array([-2.0, 2.0])  # Shoulder rotation
SP_bounds = np.array([-1.570, 1.570])   # Shoulder pitch
E_bounds  = np.array([-1.580, 1.580])  # Elbow
WP_bounds = np.array([-1.570, 1.570])   # Wrist pitch
WR_bounds = np.array([-3.142, 3.142])  # Wrist roll
G_bounds  = np.array([-0.2, 2.000])   # Gripper

# Rotation helper functions
def Rotx(theta_rad):
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

def Roty(theta_rad):
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

def Rotz(theta_rad):
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def Rot(roll, pitch, yaw):
    """General rotation"""
    return Rotz(yaw) @ Roty(pitch) @ Rotx(roll)

# --- Validation Checks ---
checks = [
    (Rot(0, 0, np.pi), R_B_W_static, "1st (World to Base)"),
    (Rot(0, 0, 0), R_S_B_static, "2nd (Base to Shoulder)"),
    (Rot(0, -np.pi/2, 0), R_UA_S_static, "3rd (Shoulder to Upper Arm)"),
    (Rot(0, 0, 0), R_LA_UA_static, "4th (Upper Arm to Lower Arm)"),
    (Rot(0, 0, np.pi/2), R_WR_LA_static, "5th (Lower Arm to Wrist)"),
    (Rot(0, 0, -np.pi/2), R_G_WR_static, "6th (Wrist to Gripper)"),
    (Rot(0, 0, 0), R_GC_G_static, "7th (Gripper to Center)"),
    (Rot(np.pi/2, np.pi, 0), R_J_G_static, "8th (Gripper to Jaw)")
]

for calculated, target, name in checks:
    if not np.allclose(calculated, target, atol=1e-4):
        print(f"{name} Rotation Matrix BAD")
        print("Target:\n", target)
        print("Calculated:\n", calculated, "\n")
    else:
        print(f"{name} check PASSED")


def get_homogenous_transform(rotation, translation):
    # This function calls np.block to assemble the sub-matrices
    # Bottom row must be 1x4: [0, 0, 0, 1]
    bottom_row = np.array([[0, 0, 0, 1]])
    
    # Assemble the 4x4 matrix
    T = np.block([
        [rotation, translation],
        [bottom_row]
    ])
    return T


# --- 6. Creating all Homogenous Transform Matrices ---
T_B_W_static   = get_homogenous_transform(R_B_W_static,   P_BORG_W)
T_S_B_static   = get_homogenous_transform(R_S_B_static,   P_SORG_B)
T_UA_S_static  = get_homogenous_transform(R_UA_S_static,  P_UAORG_S)
T_LA_UA_static = get_homogenous_transform(R_LA_UA_static, P_LAORG_UA)
T_WR_LA_static = get_homogenous_transform(R_WR_LA_static, P_WRORG_LA)
T_G_WR_static  = get_homogenous_transform(R_G_WR_static,  P_GORG_WR)
T_GC_G_static  = get_homogenous_transform(R_GC_G_static,  P_GCORG_G)
T_J_G_static   = get_homogenous_transform(R_J_G_static,   P_JORG_G)

static_transforms = [T_B_W_static, T_S_B_static, T_UA_S_static, T_LA_UA_static, T_WR_LA_static, T_G_WR_static, T_GC_G_static, T_J_G_static]

T_GC_W_zeroed = T_B_W_static @ T_S_B_static @ T_UA_S_static @ T_LA_UA_static @ T_WR_LA_static @ T_G_WR_static @ T_GC_G_static


def fwd_kinematics(thetas):
    '''
    Takes in vector of theta positions: theta1 = shoulder roll
    theta2 = soulder pitch
    etc. etc. up to the jaw
    '''
    T_links = [static_transforms[0]] #define as T_B_W_static
    #goal of the for loop is to create the transform that takes into account the current angle of the joint, and combine it with the static homotransform
    for i, theta in enumerate(thetas): #enumerate starts at 0
        R_joint_i = Rotz(theta) # Rotation matrix around z axis 
        T_joint_i = get_homogenous_transform(R_joint_i, np.zeros((3,1))) #Create homogenous transform from the rotation matrix.
        #First multiplication of the vector is with the rotation in the current frame of reference(child), then express the position in the parent frame.
        T_link_i = static_transforms[i+1] @ T_joint_i #i+1 bcs the first static transform is the world-base transform, which does not correspond to any joint
        
        #print(analytical matrices)
        if(i+1 != 6):
            T_link_i_analytical = static_transforms[i+1] @ R_z #to see them analytically
            print("\n")
            pprint(T_link_i_analytical)
            print("\n")

        T_links.append(T_link_i)

    T_G_W = np.identity(4)

    #compute the homogenous transform matrix that moves from Gripper Center frame to World frame 
    for T in T_links:
        T_G_W = T_G_W @ T
    
    return T_G_W


theta = symbols('theta')

# Define a 4x4 Z-axis rotation matrix
R_z = Matrix([
    [cos(theta), -sin(theta), 0, 0],
    [sin(theta),  cos(theta), 0, 0],
    [0,           0,          1, 0],
    [0,           0,          0, 1]
])

# Checks the zero'th position 
#if np.allclose(fwd_kinematics(np.array([0,0,0,0,0,0])), T_GC_W_zeroed, atol=1e-4):
#    print("EUREKA")
#    print(fwd_kinematics(np.array([0,0,0,0,0,0])))
#    print(" ")
#    print(T_GC_W_zeroed)
    
#Test
fwd_kinematics(np.array([0,0,0,0,0,0]))

def get_translation(T):
    """Helper function to extract X, Y, Z coordinates from a 4x4 matrix."""
    return T[0, 3], T[1, 3], T[2, 3]

def plot_named_robot_zero_pose():
    """Plots the robot and labels every joint based on the kinematic chain."""
    
    # 1. Calculate global transforms sequentially
    T_world = np.eye(4) # World origin [0,0,0]
    T_base = T_world @ T_B_W_static
    T_shoulder = T_base @ T_S_B_static
    T_upper_arm = T_shoulder @ T_UA_S_static
    T_lower_arm = T_upper_arm @ T_LA_UA_static
    T_wrist = T_lower_arm @ T_WR_LA_static
    T_gripper = T_wrist @ T_G_WR_static
    
    # Both of these branch off from the Gripper frame
    T_gripper_center = T_gripper @ T_GC_G_static
    T_jaw = T_gripper @ T_J_G_static

    # 2. Define the main kinematic chain
    main_chain = [
        (T_world, "World"),
        (T_base, "Base"),
        (T_shoulder, "Shoulder"),
        (T_upper_arm, "Upper Arm"),
        (T_lower_arm, "Lower Arm"),
        (T_wrist, "Wrist"),
        (T_gripper, "Gripper")
    ]

    # 3. Define the end-effector branches
    branches = [
        (T_gripper_center, "Gripper Center"),
        (T_jaw, "Jaw")
    ]

    # Initialize plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 4. Plot Main Chain and add Labels
    x_main, y_main, z_main = [], [], []
    for T, name in main_chain:
        x, y, z = get_translation(T)
        x_main.append(x)
        y_main.append(y)
        z_main.append(z)
        
        # Add the text label slightly offset
        ax.text(x, y, z, f"  {name}", color='black', fontsize=9, fontweight='bold', zorder=10)

    # Draw the main links
    ax.plot(x_main, y_main, z_main, '-o', color='#2ca02c', markersize=8, linewidth=4)

    # 5. Plot Branches and add Labels
    gx, gy, gz = get_translation(T_gripper)
    for T, name in branches:
        bx, by, bz = get_translation(T)
        
        # Draw the branch link from gripper to end-effector
        ax.plot([gx, bx], [gy, by], [gz, bz], '-o', color='#1f77b4', markersize=6, linewidth=3)
        ax.text(bx, by, bz, f"  {name}", color='#1f77b4', fontsize=9, fontweight='bold', zorder=10)

    # 6. Formatting
    ax.set_xlabel('X Axis (meters)')
    ax.set_ylabel('Y Axis (meters)')
    ax.set_zlabel('Z Axis (meters)')
    ax.set_title('Named Joint Visualization (Zero Pose)')
    
    # Set view angle to roughly match the screenshot perspective
    ax.view_init(elev=20, azim=45)
    
    plt.show()

# Run the plot
#plot_named_robot_zero_pose()

import itertools

def generate_workspace(steps=10):
    print(f"Generating workspace with {steps} steps per joint...")
    
    # 1. Create linear spaces for each joint bound
    sr_vals = np.linspace(-2.0, 2.0, steps)
    sp_vals = np.linspace(-1.570, 1.570, steps)
    e_vals  = np.linspace(-1.580, 1.580, steps)
    wp_vals = np.linspace(-1.570, 1.570, steps)
    wr_vals = np.linspace(-3.142, 3.142, steps)
    
    # Note: We omit the gripper bounds (G_bounds) because opening/closing 
    # the gripper does not change the position of the gripper center.
    
    workspace_points = []
    
    # 2. Sweep through all combinations
    # itertools.product gives us every combination of the joint angles
    total_iterations = steps ** 5
    count = 0
    
    for thetas in itertools.product(sr_vals, sp_vals, e_vals, wp_vals, wr_vals):
        # Calculate FK for this specific combination of angles
        T = fwd_kinematics(thetas)
        
        # Extract the x, y, z position (first 3 rows, 4th column)
        pos = T[0:3, 3]
        workspace_points.append(pos)
        
        # Simple progress tracker
        count += 1
        if count % 2000 == 0:
            print(f"Processed {count}/{total_iterations} points...")

    return np.array(workspace_points)

def plot_workspace(points):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Scatter plot: X, Y, Z coordinates
    # s=1 makes the points small, alpha=0.3 makes them slightly transparent
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, alpha=0.3, c='blue')
    
    ax.set_title("Robot Workspace")
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.set_zlabel("Z (meters)")
    
    # Keep the aspect ratio equal so the workspace doesn't look stretched
    ax.set_box_aspect([1,1,1]) 
    
    plt.show()

# --- Run the code ---
#workspace_pts = generate_workspace(steps=7)
#plot_workspace(workspace_pts)