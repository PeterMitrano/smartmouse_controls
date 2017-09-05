
# coding: utf-8

# # Jacobian Linearization Example
# 
# In this example, we will investigate how accurate a simulation using jacobian linearization is compared to the true nonlinear system dynamics. The system in this scenario is a water mixing tank. There is a hot water source and a cold water source and the goal is to acheive water of a specific temperature. The following nonlinear equations govern the system:
# 
# \begin{equation}
# \begin{split}
# \dot{h}(t) &= \frac{1}{A_T}(q_C(t) + q_H(t) - C_DA_o\sqrt{2gh(t})) \\
# \dot{T}_T(t) &= \frac{1}{h(t)A_T}(q_C(t)[T_C - T_T(t)] + q_H(t)[T_H - T_T(t)]) \\
# \end{split}
# \end{equation}
# 
# $h(t)$ is the height of the tank, $q_C$ is the flow rate of the cold water, $q_H$ is the flow rate of the hot water, These flow rates are our controls. $A_T$, $C_D$, $g$, and $A_o$ are just some constants.
# 
# Let's define some constraints for our simulation.
# 
# \begin{equation}
# \begin{split}
# h(0) &= 1.10 \\
# T_T(0) &= 81.5 \\
# q_C(0) &= \begin{cases} 0.022 & 0 \leq t \leq 25 \\
#                        0.043 & 25 \leq t \leq 100 \\ \end{cases} \\
# q_H(0) &= \begin{cases} 0.14 & 0 \leq t \leq 60 \\
#                        0.105 & 60 \leq t \leq 100 \\ \end{cases} \\
# \end{split}
# \end{equation}
# 
# We picked these numbers because they're reasonably close to the equilibrium condition of $h=1, T_T=75$. Of course the control inputs are totally arbitrary, we just want to see the impact they have on height and water temperature.
# 
# For the true system dynamics, the simulation is simple. Given the starting variable, calculate $h(0)$ and $T_T(0)$. Then, use these new outputs as the inputs at the next time step.

# In[18]:

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.formatter.useoffset'] = False
mpl.rcParams['figure.figsize'] = (10,10)


# In[25]:

def nonlinear_simulate(q_C, q_H, h0, T0, T_C=10, T_H=90, A_o=0.05, A_T=3, C_D=0.7, g=9.8, debug=False):
    h_t = h0
    T_T_t = T0
    H = []
    T = []
    for t in range(100):
        q_C_t = q_C(t)
        q_H_t = q_H(t)
        if debug:
            print(h_t, T_T_t, q_C_t, q_H_t)
        T_T_t += (q_C_t * (T_C - T_T_t) + q_H_t * (T_H - T_T_t)) / (h_t * A_T) 
        h_t += (q_C_t + q_H_t - C_D * A_o * np.sqrt(2 * g * h_t)) / A_T
        H.append(h_t)
        T.append(T_T_t)
        
    return H, T

def linear_simulate(q_C, q_H, h0, T0, h_e=1, T_e=75, T_C=10, T_H=90, A_o=0.05, A_T=3, C_D=0.7, g=9.8, debug=False):
    # first we calculate our control inputs for the input equilibrium point (h_e, T_e)
    q_C_e = (C_D*A_o*np.sqrt(2*g*h_e)*(T_H-T_e))/(T_H-T_C)
    q_H_e = (C_D*A_o*np.sqrt(2*g*h_e)*(T_e-T_C))/(T_H-T_C)
    if debug:
        print(q_C_e, q_H_e)

    # then we solve for A and B, the coefficients of our linear system
    A = np.array([[-g*C_D*A_o/(A_T*np.sqrt(2*g*h_e)), 0],[-(q_C_e*(T_C-T_e) + q_H_e*(T_H-T_e))/(h_e*h_e*A_T), -(q_C_e+q_H_e)/(h_e*A_T)]])
    B = np.array([[1/A_T,1/A_T],[(T_C-T_e)/(h_e*A_T), (T_H-T_e)/(h_e*A_T)]])
    if debug:
        print(A)
        print(B)
                
    # compute initial distance between current state and equilibrium state, our delta_x
    delta_h_t = h0 - h_e
    delta_T_t = T0 - T_e
        
    h_t = h0
    T_T_t = T0
    H = []
    T = []
    for t in range(100):
        q_C_t = q_C(t)
        q_H_t = q_H(t)

        # compute distance between our current control and equilibrium control, our delta_u
        delta_q_C = q_C_t - q_C_e
        delta_q_H = q_H_t - q_H_e

        # use the linear form to update our delta values
        delta_x = np.array([delta_h_t, delta_T_t])
        delta_u = np.array([delta_q_C, delta_q_H])
        delta_x += A@delta_x + B@delta_u
        
        delta_h_t = delta_x[0]
        delta_T_t = delta_x[1]
        
        h_t = h_e + delta_h_t
        T_T_t = T_e + delta_T_t
            
        H.append(h_t)
        T.append(T_T_t)
        
    return H, T

def q_C(t):
    if t <= 25:
        return 0.022
    else:
        return 0.043
    
def q_H(t):
    if t <= 60:
        return 0.14
    else:
        return 0.105
    
def q_C_eq(t):
    return 0.029
    
def q_H_eq(t):
    return 0.126
    
def make_plots(q_C, q_H, h0, T0):
    actual_H, actual_T = nonlinear_simulate(q_C, q_H, h0, T0)
    linear_H, linear_T = linear_simulate(q_C, q_H, h0, T0)

    fig, ax = plt.subplots(nrows=2, ncols=1)
    fig.tight_layout()
    ax[0].plot(actual_H, label="actual")
    ax[0].plot(linear_H, label="linear", linestyle='dashed')
    ax[0].set_title("Water Height: Actual vs Linearization")
    ax[0].set_ylabel("Meters")
    ax[0].legend()

    ax[1].plot(actual_T, label="actual")
    ax[1].plot(linear_T, label="linear", linestyle='dashed')
    ax[1].set_title("Water Temp: Actual vs Linearization")
    ax[1].set_ylabel("Degrees")
    ax[1].legend()

    plt.show()
    
make_plots(q_C, q_H, 1.1, 81.5)


# ## HELL YEA, It works!
# 
# We just ran a simulation comparison between a linearization of a nonlinear system and the original nonlinear system. Go math! Let's try some other scenarios...

# ### Higher water, lower temp

# In[27]:

make_plots(q_C, q_H, 1.5, 65)


# ### Lower Water, Higher Temp

# In[31]:

make_plots(q_C, q_H, 0.5, 95)


# ### What about the equilibrium point itself? (and use the correct control for that point)

# In[34]:

make_plots(q_C_eq, q_H_eq, 1.0, 75.0)


# ### Unsurprisingly, it matches perfectly! (and barely changes, look at the axis scale)

# In[ ]:



