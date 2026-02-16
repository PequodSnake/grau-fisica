import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# parametres fisics
L = 0.02
k = 0.56
rho = 1081.0
cv = 3686.0
Pext = 9.44e5
Tref = 309.65
deltaT = 43.5
Ttarget = 353.15
Tsafe = 323.15
Ttarget_C = Ttarget - 273.15
Tsafe_C = Tsafe - 273.15
t0 = (L**2 * rho * cv) / k

C = Pext * L**2 / (k * deltaT)

# parametres numerics
Nx = 101
dx = L / (Nx - 1)
dx_star = dx / L
dt_star = dx_star**2
dt = dt_star * t0
max_steps = 2000000

r = dt_star / (2 * dx_star**2)

theta = np.zeros(Nx)
theta_prev = np.zeros(Nx)

# creem la matriu
aa = np.zeros(Nx)
bb = np.ones(Nx)
cc = np.zeros(Nx)
aa[1:Nx-1] = -r
bb[1:Nx-1] = 1 + 2*r
cc[1:Nx-1] = -r

x_vals = np.linspace(0, L, Nx)
x_vals_mm = x_vals * 1000

# zona malalta
i_sick_start = int(round(0.0075 / L * (Nx-1)))
i_sick_end   = int(round(0.0125 / L * (Nx-1)))

t_sick = 0.0

def thomas(a, b, c, d):
    n = len(d)
    cp = np.zeros(n)
    dp = np.zeros(n)
    x = np.zeros(n)

    # convertim el sistema en triangular superior
    
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    
    for i in range(1, n):
        m = b[i] - a[i] * cp[i-1]
        cp[i] = c[i] / m
        dp[i] = (d[i] - a[i]*dp[i-1]) / m
    

    # solucionem a partir de x_n
    x[-1] = dp[-1]
    for i in range(n-2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i+1]
    
    return x

# animacio
fig, ax = plt.subplots()
ax.tick_params(direction="in", top=True, bottom=True, left=True, right=True)
# linia principal que s'actualitza a cada frame
line, = ax.plot([], [], lw=2, color='blue')

# linia horitzontal de Ttarget i Tsafe en °C
target_line = ax.axhline(Ttarget_C, color='red', linestyle='--', lw=1.5, label='80°C')
safe_line = ax.axhline(Tsafe_C, color='green', linestyle='--', lw=1.5, label='50°C')

# linies verticals que marquen l'inici i final de la zona malalta en mm
sick_start_line = ax.axvline(x_vals_mm[i_sick_start], color='orange', linestyle='--', lw=1.5, label='Zona malalta')
sick_end_line = ax.axvline(x_vals_mm[i_sick_end], color='orange', linestyle='--', lw=1.5)


ax.set_xlim(0, L*1000)
ax.set_ylim(Tref-273.15, Ttarget_C + 5)
ax.set_xlabel("Posició (mm)")
ax.set_ylabel("Temperatura (°C)")
ax.legend(loc='upper right')
#ax.set_title("Evolucio de la temperatura")

# array on guardem els frames per a l'animacio
frames = []

# guardem un frame cada frame_skip passos
frame_skip = 2

# bucle temporal principal
for step in range(1, max_steps + 1):

    # guardem la solucio anterior per si s'ha d'aturar
    theta_prev[:] = theta[:]

    # construim el vector dd
    dd = np.zeros(Nx)
    dd[1:Nx-1] = r*theta[:-2] + (1 - 2*r)*theta[1:-1] + r*theta[2:] + C*dt_star

    # resolem el sistema tridiagonal
    theta = thomas(aa, bb, cc, dd)

    Tmax = np.max(theta)*deltaT + Tref

    # control 80C zona malalta
    if Tmax >= Ttarget:
        theta[:] = theta_prev[:]
        break

    Tmax_healthy = max(np.max(theta[:i_sick_start]), np.max(theta[i_sick_end+1:]))*deltaT + Tref

    # control 50C zona sana
    if Tmax_healthy >= Tsafe:
        theta[:] = theta_prev[:]
        break

    # comptem temps on la zona malalta esta entre Tsafe i Ttarget
    for i in range(i_sick_start, i_sick_end+1):
        Tloc = theta[i]*deltaT + Tref
        if Tsafe <= Tloc <= Ttarget:
            t_sick += dt
            break

    # guardem frame per animacio
    if step % frame_skip == 0:
        frames.append(theta*deltaT + Tref - 273.15)

t_dim = (step-1) * dt

# temperatures finals
Tmax_sick = np.max(theta[i_sick_start:i_sick_end+1])*deltaT + Tref
Tmax_healthy = max(np.max(theta[:i_sick_start]), np.max(theta[i_sick_end+1:]))*deltaT + Tref
Tmax_total = np.max(theta)*deltaT + Tref

print("temps total:", t_dim, "s")
print("temps amb zona malalta entre 50-80C:", t_sick, "s")
print("temperatura maxima final:", Tmax_total, "K")
print("temperatura maxima zona malalta:", Tmax_sick, "K")
print("temperatura maxima zona sana:", Tmax_healthy, "K")

# graf final
plt.figure()
plt.plot(x_vals_mm, theta*deltaT + Tref - 273.15, label="Temperatura final", color='blue')
plt.axhline(Ttarget_C, color='red', linestyle='--', lw=1.5, label="80°C")
plt.axhline(Tsafe_C, color='green', linestyle='--', lw=1.5, label="50°C")
plt.axvline(x_vals_mm[i_sick_start], color='orange', linestyle='--', lw=1.5, label="Zona malalta")
plt.axvline(x_vals_mm[i_sick_end], color='orange', linestyle='--', lw=1.5)
plt.xlabel("Posició (mm)")
plt.ylabel("Temperatura (°C)")
plt.tick_params(direction="in", top=True, bottom=True, left=True, right=True)
#plt.title("Temperatura final")
plt.legend()
plt.savefig("temperatura_final_CN.png")
plt.close()


# funcio que actualitza la linia a cada frame de l'animacio
def update(frame):
    line.set_data(x_vals_mm, frame)
    return line, target_line, safe_line, sick_start_line, sick_end_line

# generem animacio
anim = FuncAnimation(fig, update, frames=frames, blit=True, repeat=False)
anim.save("evolucio_temperatura_CN.gif", writer='pillow', fps=15)