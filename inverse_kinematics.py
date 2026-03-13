import numpy as np
import matplotlib.pyplot as plt

#link lengths
a1=1
a2=1
a3=0.5

def inverse_kinematics(x,y,z):
    #get_shoulder_roll  
    q1 = np.arctan(y/x) #TODO this assumes the shoulder rolla around the World Frame(NOT TRUE)
    # l_e is the distance of the EE in the xy plane of the World frame
    l_e  =  np.sqrt( x**2 + y**2 )

    #l1+l2+l3 = l_e; l_12=l1+l2
    #get mean_arm_angle from x y z, wrt shoulder pitch origin frame in World frame
    # mean_arm_angle = q2 + q3 + q4 
    mean_arm_angle = np.atan(z/(np.sqrt(x**2 + y**2))) #TODO this formula assumes the origin of the shoulder pitch joint is at the origin of the world frame(which is not true)
    l_12 = l_e - a3*np.cos(mean_arm_angle)
    z_12 = z - a3 * np.sin(mean_arm_angle)

    #solve linear 3 joint arm
    r = np.sqrt( l_12**2 + z_12**2)
    q2 = np.arctan( l_12 / z_12) - np.arccos((a1**2 + r**2 -a2**2) / (2*a1*r))
    q3 = np.arccos((l_12-a1*np.cos(q2)) /a2) -q2
    q4 = mean_arm_angle -q2 - q3

    return(np.array([ q1 , q2 , q3 , q4]))

q = inverse_kinematics(0.5, 0.5 , 0.5 , np.pi/6)
print(q)





