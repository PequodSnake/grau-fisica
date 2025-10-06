import numpy as np
import matplotlib.pyplot as plt

epsilon_0 = 8.854e-12 

longitud_L = 1.0
Q_L = 1e-6
Q_puntual = -1e-6
posicio_carrega_puntual = (-0.1, 0.1)  

# Definir el conductor en forma de L com un conjunt de elements de carrega discreta
num_segments = 300 
x_L1 = np.linspace(-longitud_L, 0, num_segments // 2)  # Segment horitzontal
y_L1 = np.zeros_like(x_L1)

x_L2 = np.zeros(num_segments // 2)  # Segment vertical
y_L2 = np.linspace(0, longitud_L, num_segments // 2)

x_L = np.concatenate([x_L1, x_L2])
y_L = np.concatenate([y_L1, y_L2])

carregues_L = np.full(num_segments, Q_L / num_segments)  # Carrega per segment

x_interval = np.linspace(-1.5, 0, 100)
y_interval = np.linspace(0, 1.5, 100)
X, Y = np.meshgrid(x_interval, y_interval)

# Funcio per calcular el camp electric degut a una carrega puntual
def camp_electric(q, r0, x, y):
    rx = x - r0[0]
    ry = y - r0[1]
    r = np.sqrt(rx**2 + ry**2)
    r[r == 0] = 1e-9
    Ex = (1 / (4 * np.pi * epsilon_0)) * q * rx / r**3
    Ey = (1 / (4 * np.pi * epsilon_0)) * q * ry / r**3
    return Ex, Ey

Ex_total, Ey_total = np.zeros_like(X), np.zeros_like(Y)

# Contribucio del conductor en L
for i in range(num_segments):
    Ex, Ey = camp_electric(carregues_L[i], (x_L[i], y_L[i]), X, Y)
    Ex_total += Ex
    Ey_total += Ey

# Contribucio de la carrega puntual
Ex_p, Ey_p = camp_electric(Q_puntual, posicio_carrega_puntual, X, Y)
Ex_total += Ex_p
Ey_total += Ey_p

plt.figure(figsize=(8, 8))
plt.streamplot(X, Y, Ex_total, Ey_total, color=np.sqrt(Ex_total**2 + Ey_total**2), cmap='plasma', density=2, arrowsize=0, linewidth=0.7)
plt.scatter(x_L, y_L, color='black', marker='o', label='Conductor en L')
plt.scatter(*posicio_carrega_puntual, color='red', marker='o', s=100, label='Càrrega puntual')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.legend(fontsize=16, loc='upper center', frameon=True, fancybox=True)
plt.xlim(-1.5, 0)
plt.ylim(0, 1.5)
plt.tick_params(labeltop=True, labelright=True)
plt.show()




