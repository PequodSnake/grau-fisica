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
dt_star = 0.25 * dx_star**2
dt = dt_star * t0
r = dt_star / (dx_star**2)

max_steps = 2000000

# inicialitzem camps
theta = np.zeros(Nx)
theta_prev = np.zeros(Nx)

x_vals = np.linspace(0, L, Nx)
x_vals_mm = x_vals * 1000

# zona malalta
i_sick_start = int(round(0.0075 / L * (Nx-1)))
i_sick_end   = int(round(0.0125 / L * (Nx-1)))

t_sick = 0.0

# animacio
fig, ax = plt.subplots()
ax.tick_params(direction="in", top=True, bottom=True, left=True, right=True)
# linia principal que s'actualitza a cada frame
line, = ax.plot([], [], lw=2, color='blue', label="Temperatura")

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
#ax.set_title("Evolució de la temperatura")
ax.legend(loc='upper right')

frames = []
frame_skip = 5

# bucle temporal
for step in range(1, max_steps + 1):

    theta_prev[:] = theta[:]

    theta_new = theta.copy()

    # actualització explicita
    theta_new[1:-1] = (
        theta[1:-1] +
        r*(theta[:-2] - 2*theta[1:-1] + theta[2:]) +
        C * dt_star
    )

    theta = theta_new

    Tmax = np.max(theta)*deltaT + Tref

    # control 80°C
    if Tmax >= Ttarget:
        theta[:] = theta_prev
        break

    # control zona sana 50°C
    Tmax_healthy = max(np.max(theta[:i_sick_start]),
                       np.max(theta[i_sick_end+1:]))*deltaT + Tref

    if Tmax_healthy >= Tsafe:
        theta[:] = theta_prev
        break

    # temps on la zona malalta esta entre Tsafe i Ttarget
    for i in range(i_sick_start, i_sick_end+1):
        Tloc = theta[i]*deltaT + Tref
        if Tsafe <= Tloc <= Ttarget:
            t_sick += dt
            break

    # guardar frame
    if step % frame_skip == 0:
        frames.append(theta*deltaT + Tref - 273.15)

t_dim = (step-1) * dt

Tmax_sick = np.max(theta[i_sick_start:i_sick_end+1])*deltaT + Tref
Tmax_healthy = max(np.max(theta[:i_sick_start]),
                   np.max(theta[i_sick_end+1:]))*deltaT + Tref
Tmax_total = np.max(theta)*deltaT + Tref

print("temps total:", t_dim, "s")
print("temps amb zona malalta entre 50-80C:", t_sick, "s")
print("temperatura maxima final:", Tmax_total, "K")
print("temperatura maxima zona malalta:", Tmax_sick, "K")
print("temperatura maxima zona sana:", Tmax_healthy, "K")

plt.figure()
plt.plot(x_vals_mm, theta*deltaT + Tref - 273.15, label="Temperatura final", color='blue')
plt.axhline(Ttarget_C, color='red', linestyle='--', lw=1.5, label="80°C")
plt.axhline(Tsafe_C, color='green', linestyle='--', lw=1.5, label="50°C")
plt.axvline(x_vals_mm[i_sick_start], color='orange', linestyle='--', lw=1.5, label="Zona malalta")
plt.axvline(x_vals_mm[i_sick_end], color='orange', linestyle='--', lw=1.5)
plt.xlabel("Posició (mm)")
plt.ylabel("Temperatura (°C)")
plt.tick_params(direction="in", top=True, bottom=True, left=True, right=True)
plt.legend()
plt.savefig("temperatura_final_EE.png")
plt.close()

def update(frame):
    line.set_data(x_vals_mm, frame)
    return line, target_line, safe_line, sick_start_line, sick_end_line

anim = FuncAnimation(fig, update, frames=frames, blit=True, repeat=False)
anim.save("evolucio_temperatura_EE.gif", writer='pillow', fps=15)