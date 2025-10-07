import numpy as np
from scipy.integrate import odeint

##############################################################################
# 1) Lorenz System Data Generation
##############################################################################

def lorenz_deriv(state, t, sigma=10.0, rho=28.0, beta=8.0/3.0):
    """
    Lorenz system derivatives: [dx/dt, dy/dt, dz/dt].
    """
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x*(rho - z) - y
    dzdt = x*y - beta*z
    return [dxdt, dydt, dzdt]

def generate_lorenz_data(
    initial_state=[1.0, 1.0, 1.0],
    tmax=25.0,
    dt=0.01,
    sigma=10.0,
    rho=28.0,
    beta=8.0/3.0
):
    """
    Integrates the Lorenz system from initial_state up to tmax with time step dt.
    Returns:
      t_vals: array of time points
      sol   : array shape [num_steps, 3] with [x(t), y(t), z(t)]
    """
    num_steps = int(tmax / dt)
    t_vals = np.linspace(0, tmax, num_steps)
    sol = odeint(lorenz_deriv, initial_state, t_vals, args=(sigma, rho, beta))
    return t_vals, sol


##############################################################################
# Rössler system
##############################################################################

def rossler_derivatives(state, t, a=0.2, b=0.2, c=5.7):
    """Compute time derivatives [dx/dt, dy/dt, dz/dt] for the Rössler system."""
    x, y, z = state
    dxdt = -y - z
    dydt = x + a * y
    dzdt = b + z * (x - c)
    return [dxdt, dydt, dzdt]

def generate_rossler_data(
    initial_state=[1.0, 0.0, 0.0],
    tmax=25.0,
    dt=0.01,
    a=0.2,
    b=0.2,
    c=5.7
):
    """
    Numerically integrate Rössler equations x'(t), y'(t), z'(t) using odeint.
    Returns:
       t_vals: array of time points
       sol   : array shape [num_steps, 3] of [x(t), y(t), z(t)]
    """
    num_steps = int(tmax / dt)
    t_vals = np.linspace(0, tmax, num_steps)
    sol = odeint(rossler_derivatives, initial_state, t_vals, args=(a, b, c))
    return t_vals, sol

##############################################################################
# Chen system
##############################################################################

def chen_deriv(state, t, a=35.0, b=3.0, c=28.0):
    """
    Computes derivatives [dx/dt, dy/dt, dz/dt] for Chen system:
      dx/dt = a*(y - x)
      dy/dt = (c - a)*x + c*y - x*z
      dz/dt = x*y - b*z
    """
    x, y, z = state
    dxdt = a*(y - x)
    dydt = (c - a)*x + c*y - x*z
    dzdt = x*y - b*z
    return [dxdt, dydt, dzdt]

def generate_chen_data(
    initial_state=[1.0, 1.0, 1.0],
    tmax=50.0,
    dt=0.01,
    a=35.0,
    b=3.0,
    c=28.0
):
    """
    Integrates Chen's system from 'initial_state' up to time 'tmax' with step size 'dt'.
    Returns:
      t_vals: time array of length T
      sol   : array shape [T, 3], the trajectory [x(t), y(t), z(t)]
    """
    num_steps = int(tmax / dt)
    t_vals = np.linspace(0, tmax, num_steps)
    sol = odeint(chen_deriv, initial_state, t_vals, args=(a, b, c))
    return t_vals, sol



##############################################################################
# Chua system
##############################################################################

def chua_diode(x, m0=-1.143, m1=-0.714):
    """
    Piecewise-linear Chua diode:
      f(x) = m1*x + 0.5*(m0 - m1)*(|x+1| - |x-1|)
    """
    return m1*x + 0.5*(m0 - m1)*(abs(x+1) - abs(x-1))

def chua_deriv(state, t, alpha=9.0, beta=14.286, m0=-1.143, m1=-0.714):
    """
    Derivatives for the Chua system:
      dx/dt = alpha*( y - x - f(x) )
      dy/dt = x - y + z
      dz/dt = -beta*y
    """
    x, y, z = state
    fx = chua_diode(x, m0, m1)  # nonlinear diode term
    dxdt = alpha*(y - x - fx)
    dydt = x - y + z
    dzdt = -beta*y
    return [dxdt, dydt, dzdt]

def generate_chua_data(
    initial_state=[0.2, 0.0, 0.0],
    tmax=50.0,
    dt=0.01,
    alpha=9.0,
    beta=14.286,
    m0=-1.143,
    m1=-0.714
):
    """
    Integrate Chua's system from 'initial_state' up to time 'tmax' with step 'dt'.
    Returns:
      t_vals: array of time points
      sol   : array shape [num_steps, 3] of [x(t), y(t), z(t)]
    """
    num_steps = int(tmax / dt)
    t_vals = np.linspace(0, tmax, num_steps)

    sol = odeint(chua_deriv, initial_state, t_vals, args=(alpha, beta, m0, m1))
    return t_vals, sol


##############################################################################
# Mackey-Glass system
##############################################################################

def generate_mackey_glass_data(
    initial_value=0.2,
    tmax=100.0,
    dt=0.01,
    beta=0.2,
    gamma=0.1,
    tau=17
):
    """
    Generate Mackey-Glass time series data.

    Parameters:
        initial_value : float
            Initial value of x(t) for t in [0, tau]
        tmax : float
            Total simulation time
        dt : float
            Time step size
        beta, gamma : floats
            Mackey-Glass equation parameters
        tau : float
            Time delay in seconds

    Returns:
        t_vals : ndarray
            Time values
        x : ndarray
            Time series of x(t)
    """
    num_steps = int(tmax / dt)
    delay_steps = int(tau / dt)
    t_vals = np.linspace(0, tmax, num_steps)

    # Initialize x with delay
    x = np.zeros(num_steps)
    x[:delay_steps] = initial_value

    for t in range(delay_steps, num_steps):
        x_tau = x[t - delay_steps]
        x[t] = x[t - 1] + dt * (beta * x_tau / (1 + x_tau**10) - gamma * x[t - 1])

    return t_vals, x