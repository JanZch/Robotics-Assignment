import pinocchio as pin
import numpy as np
import sys

#This file is used to parametrize a square trajectory an use inverse kinematics generated from 
# the .urdf file using the Pinocchio library to output the corresponding list of joint states.
# the data is saved in a csv file and also printed in the command line and copied by hand to the
# example_pos_traj.py file (same folder) which is the position controller used to draw the square.

# 1. Configuration
urdf_path = "lerobot.urdf"
target_frame_name = "Gripper_Center_Fixed"

# Load model
model = pin.buildModelFromUrdf(urdf_path)
data = model.createData()
if not model.existFrame(target_frame_name):
    print(f"Error: '{target_frame_name}' not found.")
    sys.exit(1)
frame_id = model.getFrameId(target_frame_name)

# 2. Define the Square Waypoints (XY Plane)
center_x, center_y, z = 0.0, 0.25, 0.0
half_side = 0.05

corners = np.array([
    [center_x + half_side, center_y + half_side, z], # Top-Right
    [center_x - half_side, center_y + half_side, z], # Top-Left
    [center_x - half_side, center_y - half_side, z], # Bottom-Left
    [center_x + half_side, center_y - half_side, z], # Bottom-Right
    [center_x + half_side, center_y + half_side, z]  # Back to Top-Right
])

# Interpolate points along the edges (10 points per edge)
waypoints = []
for i in range(len(corners) - 1):
    segment = np.linspace(corners[i], corners[i+1], num=30, endpoint=False)
    waypoints.extend(segment)
waypoints.append(corners[-1]) # Add the final closing point
waypoints = np.array(waypoints)

# 3. IK Solver Function (Position Only + Joint Limits)
def solve_ik(target_pos, q_init):
    q = q_init.copy()
    
    for i in range(500):
        pin.framesForwardKinematics(model, data, q)
        
        # Translation error (3D)
        current_pos = data.oMf[frame_id].translation
        err = current_pos - target_pos
        
        if np.linalg.norm(err) < 1e-4:
            break
            
        # 3D Position-only Jacobian
        J = pin.computeFrameJacobian(model, data, q, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:3, :]
        
        # Integration
        q = pin.integrate(model, q, -0.1 * np.linalg.pinv(J) @ err)
        
        # Enforce URDF Joint Limits
        q = np.clip(q, model.lowerPositionLimit, model.upperPositionLimit)
        
    # Normalize radians to stay within [-pi, pi]
    q = np.arctan2(np.sin(q), np.cos(q))
        
    return q

# 4. Calculate Joint Positions for the Trajectory
joint_trajectory = []
q_current = pin.neutral(model) # Initial guess

for pt in waypoints:
    # Solve IK for the current waypoint, seeding with the last joint position
    q_current = solve_ik(pt, q_current) 
    joint_trajectory.append(q_current)

# 5. Output Results
print(f"Generated {len(waypoints)} waypoints for the XY square.")
print(f"\nFirst waypoint (x, y, z): \n{waypoints[0]}")
print(f"First joint configuration (rad): \n{joint_trajectory[0]}")

print(f"\nMidpoint waypoint (x, y, z): \n{waypoints[20]}")
print(f"Midpoint joint configuration (rad): \n{joint_trajectory[20]}")

# Save trajectory (values are in radians)
np.savetxt("square_joints.csv", joint_trajectory, delimiter=",")

data = np.loadtxt("square_joints.csv", delimiter=",")
for row in data:
    # Formats each row with comma-separated values
    print(f"    [{row[0]}, {row[1]}, {row[2]}, {row[3]}, {row[4]}],")
print("])")