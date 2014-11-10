import matplotlib.pyplot as plt
import numpy as np
import gary.integrate as si
from matplotlib.gridspec import GridSpec

def F(t, w, A, omega_D):
    q,p = w.T
    q_dot = p
    p_dot = -np.sin(q) + A*omega_D*np.cos(omega_D*t)
    return np.array([q_dot, p_dot]).T

integrator = si.DOPRI853Integrator(F, func_args=(0.07, 0.75))
times,qps = integrator.run([3.,0.], dt=0.1, nsteps=10000)
q = qps[:,0,0]
p = qps[:,0,1]

# plotting
gs = GridSpec(2,2)

plt.figure(figsize=(10,10))
plt.subplot(gs[0,:])
plt.plot(q, p, marker=None)
plt.xlabel("$q$")
plt.ylabel("$p$")

plt.subplot(gs[1,0])
plt.plot(times, q, marker=None)
plt.xlabel("time step")
plt.ylabel("$q$")
plt.xlim(min(times),max(times))

plt.subplot(gs[1,1])
plt.plot(times, p, marker=None)
plt.xlabel("time step")
plt.ylabel("$p$")
plt.xlim(min(times),max(times))

plt.savefig("../_static/integrate/forced-pendulum.png")
