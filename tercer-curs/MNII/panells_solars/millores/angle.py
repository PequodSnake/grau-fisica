import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# parametres
G = 6.67430e-11
Msol = 1.98847e30
mu = G * Msol
AU = 1.496e11
e = 0.0167
a = AU

r0 = a
t0 = np.sqrt(r0**3 / mu)

# condicions inicials (1 gener)
theta_phys = -0.0344
h_phys = np.sqrt(mu * a * (1 - e**2))
r_phys = a * (1 - e**2) / (1 + e * np.cos(theta_phys))
vr_phys = (mu / h_phys) * e * np.sin(theta_phys)
omega_phys = h_phys / r_phys**2

# normalitzacio [r, vr, omega, theta]
y = np.array([
    r_phys / r0,
    vr_phys * t0 / r0,
    omega_phys * t0,
    theta_phys
])

# ubicacio i panells
lat = 41.5020697248622
lon = 2.1039072409727404
lat_rad = np.radians(lat)
lon_rad = np.radians(lon)
inc = np.radians(23.44)
omega_terra = 2 * np.pi / (24 * 3600)
solar_constant = 1361.0
area_panell = 2.0
potencia_max = 400.0
num_panells = 1



# Base local al punt (lat, lon) en coordenades globals (mateix sistema que sol_dir_*)
up = np.array([
    np.cos(lat_rad) * np.cos(lon_rad),
    np.cos(lat_rad) * np.sin(lon_rad),
    np.sin(lat_rad)
])

east = np.array([
    -np.sin(lon_rad),
    np.cos(lon_rad),
    0.0
])

north = np.array([
    -np.sin(lat_rad) * np.cos(lon_rad),
    -np.sin(lat_rad) * np.sin(lon_rad),
    np.cos(lat_rad)
])

def normal_panell(tilt_rad, azimut_rad):
    """
    Normal unitària del panell en coordenades globals.

    Convencions:
      - tilt_rad: 0 = panell horitzontal; 90° = vertical
      - azimut_rad: 0=N, 90=E, 180=S, 270=O (direcció cap on mira el panell)
    """
    dir_h = np.cos(azimut_rad) * north + np.sin(azimut_rad) * east
    n = np.cos(tilt_rad) * up + np.sin(tilt_rad) * dir_h
    return n / np.linalg.norm(n)

# Panell fix (valors inicials; es podran optimitzar més avall)
tilt_deg = 30.0
azimut_deg = 180.0  # Sud

normal = normal_panell(np.radians(tilt_deg), np.radians(azimut_deg))

# RK4
def derivades(y):
    r, vr, omega, _ = y
    return np.array([
        vr,
        r * omega**2 - 1.0 / r**2,
        -2.0 * vr * omega / r,
        omega
    ])

dt = 1e-3
nsteps_dia = int(2 * np.pi / (dt * 365.25))
nsteps_total = nsteps_dia * 730

orbita_theta = [y[3]]

for step in range(nsteps_total):
    k1 = derivades(y)
    k2 = derivades(y + 0.5 * dt * k1)
    k3 = derivades(y + 0.5 * dt * k2)
    k4 = derivades(y + dt * k3)
    y += dt * (k1 + 2*k2 + 2*k3 + k4) / 6


    if (step + 1) % nsteps_dia == 0:

        orbita_theta.append(y[3])

orbita_theta = np.array(orbita_theta)

def calcular_produccio_horaria(theta_sol, normal_vec):

    declinacio = inc * np.sin(theta_sol - theta_phys - np.pi/2)


    sin_alt_max = np.sin(lat_rad) * np.sin(declinacio) + np.cos(lat_rad) * np.cos(declinacio)
    alt_max = np.arcsin(sin_alt_max)

    dt_h = 1.0 / 6.0
    hores = np.arange(0, 24, dt_h)

    # calcular angle horari i rotacio terrestre
    angles_horaris = (hores - 12) * omega_terra * 3600
    angles_rotacio = angles_horaris + lon_rad

    # calcular vector direccio al sol
    sol_dir_x = np.cos(declinacio) * np.cos(angles_rotacio)
    sol_dir_y = np.cos(declinacio) * np.sin(angles_rotacio)
    sol_dir_z = np.full_like(angles_rotacio, np.sin(declinacio))

    # calcular angle entre panell i sol
    cos_angle = sol_dir_x * normal_vec[0] + sol_dir_y * normal_vec[1] + sol_dir_z * normal_vec[2]

    # filtrar hores amb sol
    cond = cos_angle > 0
    irradiacions = np.zeros_like(cos_angle)
    potencies = np.zeros_like(cos_angle)

    # calcular irradiacio i potencia només quan el sol esta per sobre de l'horitzo
    if np.any(cond):
        irradiacio_efectiva = solar_constant * cos_angle[cond]
        irradiacions[cond] = irradiacio_efectiva
        ratio_irradiacio = np.minimum(irradiacio_efectiva / 1000.0, 1.0)
        potencies[cond] = ratio_irradiacio * potencia_max * num_panells

    # calcular energia total del dia en kwh
    energia_dia = np.sum(potencies) * dt_h / 1000

    idxs = potencies > 0
    return hores[idxs], potencies[idxs], irradiacions[idxs], energia_dia, np.degrees(alt_max)


def energia_anual(tilt_rad, azimut_rad, any_index=0):
    """Energia anual (kWh) per a un panell fix amb (tilt, azimut) constants."""
    nvec = normal_panell(tilt_rad, azimut_rad)
    start = 365 * any_index
    end = start + 365
    E = 0.0
    for i in range(start, end):
        _, _, _, e_dia, _ = calcular_produccio_horaria(orbita_theta[i], nvec)
        E += e_dia
    return E

def funcio_objectiu(x):
    # x[0] és el tilt, x[1] és l'azimut
    return -energia_anual(np.radians(x[0]), np.radians(x[1]), any_index=0)

res = minimize(funcio_objectiu, [35.0, 180.0], method='Nelder-Mead', 
               bounds=[(0, 90), (0, 360)], tol=1e-2)

best_t = res.x[0]
best_a = res.x[1]
best_E = -res.fun


tilt_deg = float(best_t)
azimut_deg = float(best_a)
normal = normal_panell(np.radians(tilt_deg), np.radians(azimut_deg))

print(f"  Tilt òptim: {tilt_deg:.0f}°")
print(f"  Azimut òptim: {azimut_deg:.0f}° (0=N, 90=E, 180=S, 270=O)")

# analisi dies
n_dies = 730

produccio_diaria = np.zeros(n_dies)
altitud_maxima_diaria = np.zeros(n_dies)

# bucle per calcular produccio i altitud per cada dia
for i in range(n_dies):
    _, _, _, energia, altitud = calcular_produccio_horaria(orbita_theta[i], normal)
    produccio_diaria[i] = energia
    altitud_maxima_diaria[i] = altitud

produccio_any1 = produccio_diaria[:365]
produccio_any2 = produccio_diaria[365:730]

# calcular el dia amb maxima produccio
dia_max = np.argmax(produccio_diaria)
energia_max = produccio_diaria[dia_max]

dia_min = np.argmin(produccio_diaria)
energia_min = produccio_diaria[dia_min]

# dades dies extrems
hores_dia_max, pot_dia_max, irr_dia_max, _, _ = calcular_produccio_horaria(orbita_theta[dia_max], normal)
hores_dia_min, pot_dia_min, irr_dia_min, _, _ = calcular_produccio_horaria(orbita_theta[dia_min], normal)

# imprimir per terminal
print(f"Ubicacio: {lat:.4f}°N, {lon:.4f}°E")
print(f"Installacio: {num_panells} panells de {area_panell}m² ({num_panells * potencia_max / 1000:.1f} kW total)")
print(f"ANY 1:")
print(f"  Produccio anual: {np.sum(produccio_any1):.2f} kWh")
print(f"  Produccio mitjana diaria: {np.mean(produccio_any1):.2f} kWh/dia")
print(f"  Dia amb mes produccio: Dia {dia_max + 1} ({energia_max:.2f} kWh)")
print(f"  Dia amb menys produccio: Dia {dia_min + 1} ({energia_min:.2f} kWh)")
print(f"ANY 2:")
print(f"  Produccio anual: {np.sum(produccio_any2):.2f} kWh")
print(f"  Produccio mitjana diaria: {np.mean(produccio_any2):.2f} kWh/dia")

# grafics

# altura sol
plt.figure(figsize=(12, 5))
dies_plot = np.arange(1, n_dies + 1)
plt.plot(dies_plot, altitud_maxima_diaria, 'r-', linewidth=2)
plt.fill_between(dies_plot, altitud_maxima_diaria, alpha=0.2, color='red')

events = [
    (80, 'Equinocci març', 'green'), (172, 'Solstici estiu', 'orange'),
    (266, 'Equinocci setembre', 'green'), (355, 'Solstici hivern', 'blue'),
    (445, 'Equinocci març', 'green'), (537, 'Solstici estiu', 'orange'),
    (631, 'Equinocci setembre', 'green'), (720, 'Solstici hivern', 'blue')
]
for dia, nom, color in events:
    label = nom if dia < 366 else None # evitar duplicar llegenda
    plt.axvline(dia, color=color, linestyle='--', alpha=0.5, linewidth=1, label=label)

plt.ylabel("Altitud solar màxima (graus)")
plt.title("Altitud màxima del Sol")
plt.axvline(365.5, color='black', linestyle='--', linewidth=2, alpha=0.5)
plt.grid(True, alpha=0.3)
plt.xlim(1, n_dies)
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('altitud_solar_angle.png', dpi=300)
plt.show()

# produccio diaria
plt.figure(figsize=(12, 5))
plt.plot(dies_plot, produccio_diaria, 'g-', linewidth=1.5)
plt.fill_between(dies_plot, produccio_diaria, alpha=0.2, color='green')
plt.scatter([dia_max+1], [energia_max], c='red', s=100, zorder=5, label=f'Max: dia {dia_max+1}')
plt.scatter([dia_min+1], [energia_min], c='blue', s=100, zorder=5, label=f'Min: dia {dia_min+1}')
plt.axvline(365.5, color='black', linestyle='--', linewidth=2, alpha=0.5)
plt.ylabel("Energia produida (kWh)")
plt.title("Producció diaria d'energia solar")
plt.grid(True, alpha=0.3)
plt.legend()
plt.xlim(1, n_dies)
plt.ylim(bottom=0)
plt.tight_layout()
plt.savefig('produccio_diaria_angle.png', dpi=300)
plt.show()

# produccio en un dia
def plot_dia_detall(dia, energia, hores, pot, irr, color_pot, color_irr, tipus):
    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.set_xlabel("Hora del dia")
    ax1.set_ylabel("Potència generada (kW)", color=color_pot)
    if len(hores) > 0:
        ax1.plot(hores, pot/1000, color=color_pot, linewidth=2.5, label='Potència')
        ax1.fill_between(hores, pot/1000, alpha=0.3, color=color_pot)
    ax1.tick_params(axis='y', labelcolor=color_pot)
    ax1.set_ylim(bottom=0)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Irradiació solar (W/m²)", color=color_irr)
    if len(hores) > 0:
        ax2.plot(hores, irr, color=color_irr, linewidth=2, linestyle='--', label='Irradiació')
    ax2.tick_params(axis='y', labelcolor=color_irr)
    ax2.set_ylim(bottom=0)

    plt.title(f"Dia {dia + 1} - {tipus} producció ({energia:.2f} kWh)")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.tight_layout()
    plt.savefig(f'potencia_dia_{tipus.lower()}_angle.png', dpi=300)
    plt.show()

# grafics dies extrems
plot_dia_detall(dia_max, energia_max, hores_dia_max, pot_dia_max, irr_dia_max, 'red', 'orange', 'Maxima')
plot_dia_detall(dia_min, energia_min, hores_dia_min, pot_dia_min, irr_dia_min, 'blue', 'cyan', 'Minima')
