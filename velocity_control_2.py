import numpy as np

def compute_jacobian(theta1, theta2, theta3, theta4, theta5):
    return np.array([[(0.11257*np.sin(theta2) + 0.0052*np.sin(theta2 + theta3) - 0.028*np.cos(theta2) - 0.1349*np.cos(theta2 + theta3) - 0.0601*np.cos(theta2 + theta3 + theta4) - 0.0306)*np.cos(theta1)
                    , (0.028*np.sin(theta2) + 0.1349*np.sin(theta2 + theta3) + 0.0601*np.sin(theta2 + theta3 + theta4) + 0.11257*np.cos(theta2) + 0.0052*np.cos(theta2 + theta3))*np.sin(theta1)
                    , (0.1349*np.sin(theta2 + theta3) + 0.0601*np.sin(theta2 + theta3 + theta4) + 0.0052*np.cos(theta2 + theta3))*np.sin(theta1)
                    , 0.0601*np.sin(theta1)*np.sin(theta2 + theta3 + theta4)
                    , 0]
                    , [(0.11257*np.sin(theta2) + 0.0052*np.sin(theta2 + theta3) - 0.028*np.cos(theta2) - 0.1349*np.cos(theta2 + theta3) - 0.0601*np.cos(theta2 + theta3 + theta4) - 0.0306)*np.sin(theta1)
                    , -(0.028*np.sin(theta2) + 0.1349*np.sin(theta2 + theta3) + 0.0601*np.sin(theta2 + theta3 + theta4) + 0.11257*np.cos(theta2) + 0.0052*np.cos(theta2 + theta3))*np.cos(theta1)
                    , -(0.1349*np.sin(theta2 + theta3) + 0.0601*np.sin(theta2 + theta3 + theta4) + 0.0052*np.cos(theta2 + theta3))*np.cos(theta1)
                    , -0.0601*np.sin(theta2 + theta3 + theta4)*np.cos(theta1)
                    , 0]
                    , [0
                    , -0.11257*np.sin(theta2) - 0.0052*np.sin(theta2 + theta3) + 0.028*np.cos(theta2) + 0.1349*np.cos(theta2 + theta3) + 0.0601*np.cos(theta2 + theta3 + theta4)
                    , -0.0052*np.sin(theta2 + theta3) + 0.1349*np.cos(theta2 + theta3) + 0.0601*np.cos(theta2 + theta3 + theta4)
                    , 0.0601*np.cos(theta2 + theta3 + theta4)
                    , 0]])

# Pseudonvert

# Apply velocity vector to get joint velocities


