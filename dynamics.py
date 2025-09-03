import numpy as np

def State_dynamics(positions, step, time_step_delta, final_time, control_input, mean, covariance, time):

    # retrieve the positions
    x1, x2, x3, x4, x5 = positions

    # Compute the drift vector field
    f1 = np.sqrt(np.abs(x3))*x4 + np.sin(x1) + (x5**2)*x2
    f2 = 1.5*(x3**(2))*x5 + np.cos(x4+x3) + x1*np.sqrt(np.abs(x2))*np.sin(x3)
    f3 = x5**2 - (x4**2)*(x3**3)
    f4 = (x1*x3-x2)**(3)
    f5 = -x1*x5
    
    # f1 = -x1 + np.cos(x2)
    # f2 = -x2 + np.sin(x3)
    # f3 = -x3 + np.cos(x4)
    # f4 = -x4 + np.sin(x5)
    # f5 = -x2 + np.cos(x1)
    
    f = np.array([f1, f2, f3, f4, f5]) 
    
    # Compute the diffusion matrix
    g2_11 = x1*np.cos(x2)
    g2_12 = 1 - x3*np.cos(x4)
    g2_21 = x5*x3
    g2_22 = (x4**2)*(np.sin(x2)**2)
    g2_31 = x1**2
    g2_32 = x3*np.cos(x1*x2)
    g2_41 = (x2+x1)**3 -  np.sin(x3) 
    g2_42 = 1 - x3**2
    g2_51 = x2*(np.sin(x3))**2
    g2_52 = -x5 + x1*(x4**2)
    
    g2 =  np.array([[g2_11, g2_12], 
                    [g2_21, g2_22], 
                    [g2_31, g2_32], 
                    [g2_41, g2_42], 
                    [g2_51, g2_52]])
    
    # Compute the time-varying noise matrix
    sig_11 = np.sin(time)**2
    sig_12 = 0
    sig_21 = 0
    sig_22 = np.exp(-time)
    
    Sigma = covariance*np.array([[sig_11, sig_12],
                      [sig_21, sig_22]])
    
    # Generate the Brownian motion path
    np.random.seed(0)
    dw = np.random.normal(mean, np.sqrt(time_step_delta), (2,))
    dynamics = (f + control_input)*time_step_delta + g2 @ Sigma @ dw

    return dynamics

def target_dynamics(positions, step, time):
    # retrieve the positions
    xd1, xd2, xd3, xd4, xd5 = positions
    
    xd1Dot = np.sin(2*time)
    xd2Dot = -np.cos(time)
    xd3Dot = np.sin(3*time) + np.cos(-2*time)
    xd4Dot = np.sin(time) - np.cos(-0.5*time)
    xd5Dot = np.sin(-time)
    
    return np.array([xd1Dot, xd2Dot, xd3Dot, xd4Dot, xd5Dot])

def integrate(step, positions, velocities, time_step_delta):
    positions[:, step] = positions[:, step-1] + time_step_delta * velocities[:, step]