import numpy as np
import matplotlib.pyplot as plt

# parametres fisics
L = 0.02
k = 0.56
rho = 1081
cv = 3686
Pext = 9.44e5
Tref = 309.65
deltaT = 43.5
t0 = (L*L*rho*cv)/k

C = Pext*L*L/(k*deltaT)

# parametres numerics
Nx = 101
x = np.linspace(0.0, L, Nx)
x_star = x / L
dx_star = 1.0 / (Nx-1)
t_star = 0.025

# solucio analitica
def analitica(x_star, t_star, C, k_max=200):
    f = np.zeros_like(x_star)
    for j in range(k_max+1):
        n = 2*j + 1
        coef = (4*C)/( (n**3)*(np.pi**3) )
        f += coef*(1 - np.exp(-(n**2)*(np.pi**2)*t_star))*np.sin(n*np.pi*x_star)
    return Tref + deltaT*f

T_analitica = analitica(x_star, t_star, C)

# metode tridiagonal
def thomas(a, b, c, d):
    n = len(b)
    cp = np.empty(n)
    dp = np.empty(n)
    x = np.empty(n)
    cp[0] = c[0]/b[0]; dp[0] = d[0]/b[0]
    for i in range(1,n):
        m = b[i] - a[i]*cp[i-1]
        cp[i] = c[i]/m if i<n-1 else 0.0
        dp[i] = (d[i]-a[i]*dp[i-1])/m
    x[-1] = dp[-1]
    for i in range(n-2,-1,-1):
        x[i] = dp[i]-cp[i]*x[i+1]
    return x

# metodes numerics
def crank_nicolson(dt_star):
    r = dt_star/(2*dx_star**2)
    dt = dt_star*t0
    theta = np.zeros(Nx)
    a = np.zeros(Nx); b = np.ones(Nx); c = np.zeros(Nx)
    for i in range(1,Nx-1):
        a[i]=-r; b[i]=1+2*r; c[i]=-r
    step=1
    while (step-1)*dt < t_star*t0:
        theta_prev = theta.copy()
        d = np.zeros(Nx)
        d[0]=0.0; d[-1]=0.0
        for i in range(1,Nx-1):
            d[i] = r*theta_prev[i-1] + (1-2*r)*theta_prev[i] + r*theta_prev[i+1] + C*dt_star
        theta = thomas(a,b,c,d)
        theta[0]=0.0; theta[-1]=0.0
        step+=1
    return Tref + deltaT*theta

def euler_implicit(dt_star):
    alpha = dt_star/(dx_star**2)
    dt = dt_star*t0
    theta = np.zeros(Nx)
    a = np.zeros(Nx); b = np.ones(Nx); c = np.zeros(Nx)
    for i in range(1,Nx-1):
        a[i]=-alpha; b[i]=1+2*alpha; c[i]=-alpha
    step=1
    while (step-1)*dt < t_star*t0:
        d = np.zeros(Nx); d[0]=0; d[-1]=0
        for i in range(1,Nx-1):
            d[i] = theta[i] + C*dt_star
        theta = thomas(a,b,c,d)
        theta[0]=0; theta[-1]=0
        step+=1
    return Tref + deltaT*theta

def euler_explicit(dt_star):
    r_ex = dt_star/(dx_star**2)
    dt = dt_star*t0
    theta = np.zeros(Nx)
    theta_new = np.zeros_like(theta)
    step=1
    while (step-1)*dt < t_star*t0:
        for i in range(1,Nx-1):
            theta_new[i] = theta[i] + r_ex*(theta[i-1]-2*theta[i]+theta[i+1]) + C*dt_star
        theta_new[0]=0; theta_new[-1]=0
        theta, theta_new = theta_new, theta
        step+=1
    return Tref + deltaT*theta


results = []

# CN i Euler implicit: dt* = dx_star^2 i 0.5*dx_star^2
dt_choices = [dx_star**2, 0.5*dx_star**2]
for dt_ch in dt_choices:
    T_cn = crank_nicolson(dt_ch)
    T_ei = euler_implicit(dt_ch)
    err_cn = np.abs(T_cn-T_analitica)
    err_ei = np.abs(T_ei-T_analitica)
    results.append( ('CN dt*={:.2f} (dx*)²'.format(dt_ch/dx_star**2), err_cn) )
    results.append( ('Euler implícit dt*={:.2f} (dx*)²'.format(dt_ch/dx_star**2), err_ei) )

# Euler explicit: dt* = 0.49*dx_star^2 i 0.25*dx_star^2
fe_dt_choices = [0.49*dx_star**2, 0.25*dx_star**2]
for dt_ch in fe_dt_choices:
    T_ee = euler_explicit(dt_ch)
    err_ee = np.abs(T_ee-T_analitica)
    results.append( ('Euler explícit dt*={:.2f} (dx*)²'.format(dt_ch/dx_star**2), err_ee) )


plt.figure(figsize=(10,6))
for label, err in results:
    if "CN dt*=1.00" in label:  # detectem CN amb dt* = dx_star² pq no es solapa amb una altra barra
        plt.plot(x, err, label=label, linestyle='dotted', linewidth=2)
    else:
        plt.plot(x, err, label=label)
    print(f"{label} --> Error maxim = {np.max(err):.3e} K")
plt.yscale('log')
plt.minorticks_on()
plt.xlabel('x (m)')
plt.xlim(0, L)
plt.ylabel('Error absolut (K)')
plt.tick_params(which='both',direction="in", top=True, bottom=True, left=True, right=True)
plt.legend()
plt.savefig("error_log.png", dpi=300, bbox_inches='tight')
plt.close()