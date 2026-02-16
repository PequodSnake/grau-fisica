import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.optimize import minimize

# parametres
G = 6.67430e-11
Msol = 1.98847e30
Mterra = 5.972e24
Mlluna = 7.348e22
mu_sb = G * (Msol + Mterra + Mlluna) # mu sistema solar
mu_tl = G * (Mterra + Mlluna) # mu sistema terra-lluna

AU = 1.496e11
e_sb = 0.0167 # excentricitat terra-sol
a_sb = AU

a_tl = 384400e3 # semieix major terra-lluna
e_tl = 0.0549 # excentricitat lluna

# factors de normalitzacio sistema sol-bariocentre
r0_sb = a_sb
t0_sb = np.sqrt(r0_sb**3 / mu_sb)

# factors de normalitzacio sistema terra-lluna
r0_tl = a_tl
t0_tl = np.sqrt(r0_tl**3 / mu_tl)

# condicions inicials bariocentre (1 gener)
theta_phys_sb = -0.0344
h_phys_sb = np.sqrt(mu_sb * a_sb * (1 - e_sb**2))
r_phys_sb = a_sb * (1 - e_sb**2) / (1 + e_sb * np.cos(theta_phys_sb))
vr_phys_sb = (mu_sb / h_phys_sb) * e_sb * np.sin(theta_phys_sb)
omega_phys_sb = h_phys_sb / r_phys_sb**2

# condicions inicials lluna (relatiu a terra)
# suposem que comenca al perigeu
theta_phys_tl = 0.0
h_phys_tl = np.sqrt(mu_tl * a_tl * (1 - e_tl**2))
r_phys_tl = a_tl * (1 - e_tl**2) / (1 + e_tl * np.cos(theta_phys_tl))
vr_phys_tl = (mu_tl / h_phys_tl) * e_tl * np.sin(theta_phys_tl)
omega_phys_tl = h_phys_tl / r_phys_tl**2

# normalitzacio [r, vr, omega, theta]
# sistema sol-bariocentre
y_sb = np.array([
    r_phys_sb / r0_sb, 
    vr_phys_sb * t0_sb / r0_sb, 
    omega_phys_sb * t0_sb, 
    theta_phys_sb
])

# sistema terra-lluna
y_tl = np.array([
    r_phys_tl / r0_tl, 
    vr_phys_tl * t0_tl / r0_tl, 
    omega_phys_tl * t0_tl, 
    theta_phys_tl
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

# dades climatiques (valles occidental)
probs_sol_mes = [0.55, 0.60, 0.65, 0.60, 0.65, 0.75, 0.85, 0.80, 0.70, 0.60, 0.55, 0.50]
temps_max_mes = [14.8, 18, 17.6, 20, 23, 27.3, 31.6, 31.9, 25.2, 22.1, 18.2, 14.1]
temps_min_mes = [3.9, 5.3, 6.6, 7.6, 10.1, 15, 17.9, 19, 13.7, 12.3, 8.5, 2.9]
# coeficient de perdua per temperatura
coef_temp = 0.004 


# definim el vector vertical que apunta directament cap al cel des de la nostra posicio
up = np.array([
    np.cos(lat_rad) * np.cos(lon_rad),
    np.cos(lat_rad) * np.sin(lon_rad),
    np.sin(lat_rad)
])

# establim la direccio est que es perpendicular a l'eix de rotacio de la terra
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

# calculem el vector normal unitari del panell en coordenades globals
def normal_panell(tilt_rad, azimut_rad):
    # tilt_rad: 0 = panell horitzontal; 90 = vertical
    # azimut_rad: 0=n, 90=e, 180=s, 270=o (direccio cap on mira el panell)
    
    dir_h = np.cos(azimut_rad) * north + np.sin(azimut_rad) * east
    n = np.cos(tilt_rad) * up + np.sin(tilt_rad) * dir_h
    return n / np.linalg.norm(n)

# definim valors inicials
tilt_deg = 30.0
azimut_deg = 180.0

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

orbita_theta_terra = [] # theta real de la terra

mass_ratio = Mlluna / (Mterra + Mlluna)

for step in range(nsteps_total):
    # integrem sistema sol-bariocentre
    k1 = derivades(y_sb)
    k2 = derivades(y_sb + 0.5 * dt * k1)
    k3 = derivades(y_sb + 0.5 * dt * k2)
    k4 = derivades(y_sb + dt * k3)
    y_sb += dt * (k1 + 2*k2 + 2*k3 + k4) / 6
    
    # integrem sistema terra-lluna
    # hem de cambiar el dt perque els temps caracteristics son diferents
    dt_tl = dt * (t0_sb / t0_tl)
    
    k1_m = derivades(y_tl)
    k2_m = derivades(y_tl + 0.5 * dt_tl * k1_m)
    k3_m = derivades(y_tl + 0.5 * dt_tl * k2_m)
    k4_m = derivades(y_tl + dt_tl * k3_m)
    y_tl += dt_tl * (k1_m + 2*k2_m + 2*k3_m + k4_m) / 6
    
    # cada vegada que hem completat un dia sencer
    if (step + 1) % nsteps_dia == 0:
        
        # coordenades polars a cartesianes bariocentre
        r_b = y_sb[0] * r0_sb
        th_b = y_sb[3]
        x_b = r_b * np.cos(th_b)
        y_b = r_b * np.sin(th_b)
        
        # coordenades polars a cartesianes lluna relativa terra
        r_m = y_tl[0] * r0_tl
        th_m = y_tl[3]
        x_rel = r_m * np.cos(th_m)
        y_rel = r_m * np.sin(th_m)
        
        # posicio absoluta terra
        # terra esta a -mass_ratio * r_rel del bariocentre
        x_t = x_b - mass_ratio * x_rel
        y_t = y_b - mass_ratio * y_rel
        
        # calculem theta real de la terra respecte al sol
        theta_real = np.arctan2(y_t, x_t)
        orbita_theta_terra.append(theta_real)

orbita_theta_terra = np.array(orbita_theta_terra)

def calcular_produccio_horaria(theta_sol, dia_simulacio, normal_vec, aplicar_clima=True):
    # determinar mes (0-11)
    dia_any = dia_simulacio % 365
    mes = int(dia_any / 30.44) % 12
    
    factor_nuvol = 1.0
    
    if aplicar_clima:
        # simulacio de nuvols
        rand_val = random.random()
        prob_sol = probs_sol_mes[mes]
        
        if rand_val > prob_sol:
            # dia no assolellat
            if rand_val > prob_sol + (1 - prob_sol) * 0.7:
                # molt ennuvolat
                factor_nuvol = random.uniform(0.1, 0.3)
            else:
                # variable
                factor_nuvol = random.uniform(0.4, 0.8)
    
    # temperatures del dia (amb un petit soroll aleatori)
    t_max = temps_max_mes[mes]
    t_min = temps_min_mes[mes]
    if aplicar_clima:
        t_max += random.uniform(-2, 2)
        t_min += random.uniform(-2, 2)

    # declinacio solar
    declinacio = inc * np.sin(theta_sol - theta_phys_sb - np.pi/2)

    # altitud maxima
    sin_alt_max = np.sin(lat_rad) * np.sin(declinacio) + np.cos(lat_rad) * np.cos(declinacio)
    alt_max = np.arcsin(sin_alt_max)
    
    dt_h = 1.0 / 6.0
    hores = np.arange(0, 24, dt_h)
    
    # calcular angle horari i rotacio terrestre
    angles_horaris = (hores - 12) * omega_terra * 3600
    angles_rotacio = angles_horaris + lon_rad

    # calculem la posicio vertical del sol respecte a l'horitzo
    sin_alt = (
    np.sin(lat_rad) * np.sin(declinacio) +
    np.cos(lat_rad) * np.cos(declinacio) * np.cos(angles_horaris)
    )

    # calcular vector direccio al sol
    sol_dir_x = np.cos(declinacio) * np.cos(angles_rotacio)
    sol_dir_y = np.cos(declinacio) * np.sin(angles_rotacio)
    sol_dir_z = np.full_like(angles_rotacio, np.sin(declinacio))
    
    # calcular angle entre panell i sol
    cos_angle = sol_dir_x * normal_vec[0] + sol_dir_y * normal_vec[1] + sol_dir_z * normal_vec[2]
    
    # comprovem que el sol estigui per sobre de l'horitzo (sin_alt > 0)
    # comprovem que el sol quedi per davant de la cara del panell (cos_angle > 0)
    cond = (cos_angle > 0) & (sin_alt > 0)
    irradiacions = np.zeros_like(cos_angle)
    potencies = np.zeros_like(cos_angle)
    temperatures_panel = np.zeros_like(cos_angle)
    
    if np.any(cond):
        # irradiacio base
        irradiacio_geo = solar_constant * cos_angle[cond]
        
        # aplicar factor nuvols
        irradiacio_real = irradiacio_geo * factor_nuvol
        irradiacions[cond] = irradiacio_real
        
        if aplicar_clima:
            # suposem pic de calor a les 14h
            hora_pic = 14.0
            t_ambient = t_min + (t_max - t_min) * 0.5 * (1 + np.cos((hores[cond] - hora_pic) * 2 * np.pi / 24))
            
            # temperatura del panell
            t_panel = t_ambient + 0.025 * irradiacio_real
            temperatures_panel[cond] = t_panel
            
            # calcular eficiencia termica
            perdua_temp = np.maximum(0, t_panel - 25.0) * coef_temp
            factor_termic = 1.0 - perdua_temp
            factor_termic = np.maximum(0, factor_termic) # no pot ser negatiu
            
            # potencia final
            ratio_irradiacio = np.minimum(irradiacio_real / 1000.0, 1.0)
            potencies[cond] = ratio_irradiacio * potencia_max * num_panells * factor_termic
        else:
            # sense efecte temperatura per optimitzacio
            ratio_irradiacio = np.minimum(irradiacio_real / 1000.0, 1.0)
            potencies[cond] = ratio_irradiacio * potencia_max * num_panells
    
    # calcular energia total del dia en kwh
    energia_dia = np.sum(potencies) * dt_h / 1000
    
    # pels grafics, mostrem nomes les hores on el sol es visible
    idxs = potencies > 0
    return hores[idxs], potencies[idxs], irradiacions[idxs], energia_dia, np.degrees(alt_max)

# calculem l'energia total acumulada permetent desactivar el clima
def energia_anual(tilt_rad, azimut_rad, any_index=0, aplicar_clima=True):
    nvec = normal_panell(tilt_rad, azimut_rad)
    start = 365 * any_index
    end = start + 365
    E = 0.0
    # iterem per cada dia de l'any per sumar la produccio diaria
    for i in range(start, end):
        _, _, _, e_dia, _ = calcular_produccio_horaria(orbita_theta_terra[i], i, nvec, aplicar_clima=aplicar_clima)
        E += e_dia
    return E

# definim la funcio que minimitzarem per trobar el maxim d energia
def funcio_objectiu(x):
    return -energia_anual(np.radians(x[0]), np.radians(x[1]), any_index=0, aplicar_clima=False)

# busquem els angles ideals utilitzant un algorsime en lloc de provar totes les combinacions
res = minimize(funcio_objectiu, [35.0, 180.0], method='nelder-mead')

best_t = res.x[0]
best_a = res.x[1]
best_E = -res.fun

# assignem els valors finals del tilt i l'azimut trobats
tilt_deg = float(best_t)
azimut_deg = float(best_a)
# calculem la normal definitiva del panell amb els angles optims
normal = normal_panell(np.radians(tilt_deg), np.radians(azimut_deg))

# analisi dies
n_dies = 730

produccio_diaria = np.zeros(n_dies)
altitud_maxima_diaria = np.zeros(n_dies)
dades_diaries = []

# bucle per calcular produccio i altitud per cada dia
for i in range(n_dies):
    hores, pot, irr, energia, altitud = calcular_produccio_horaria(orbita_theta_terra[i], i, normal, aplicar_clima=True)
    produccio_diaria[i] = energia
    altitud_maxima_diaria[i] = altitud
    dades_diaries.append((hores, pot, irr))

produccio_any1 = produccio_diaria[:365]
produccio_any2 = produccio_diaria[365:730]

# calcular el dia amb maxima i minima produccio
dia_max = np.argmax(produccio_diaria)
energia_max = produccio_diaria[dia_max]

dia_min = np.argmin(produccio_diaria)
energia_min = produccio_diaria[dia_min]

# dades dies extrems
hores_dia_max, pot_dia_max, irr_dia_max = dades_diaries[dia_max]
hores_dia_min, pot_dia_min, irr_dia_min = dades_diaries[dia_min]

# imprimir per terminal
print(f"Ubicacio: {lat:.4f}°N, {lon:.4f}°E")
print(f"Installacio: {num_panells} panells de {area_panell}m² ({num_panells * potencia_max / 1000:.1f} kW total)")
print(f"  Tilt optim: {tilt_deg:.0f}°")
print(f"  Azimut optim: {azimut_deg:.0f}° (0=N, 90=E, 180=S, 270=O)")
print(f"  Dia amb mes produccio: Dia {dia_max + 1} ({energia_max:.2f} kWh)")
print(f"  Dia amb menys produccio: Dia {dia_min + 1} ({energia_min:.2f} kWh)")
print(f"ANY 1:")
print(f"  Produccio anual: {np.sum(produccio_any1):.2f} kWh")
print(f"  Produccio mitjana diaria: {np.mean(produccio_any1):.2f} kWh/dia")
print(f"ANY 2:")
print(f"  Produccio anual: {np.sum(produccio_any2):.2f} kWh")
print(f"  Produccio mitjana diaria: {np.mean(produccio_any2):.2f} kWh/dia")

# grafics

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True

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
    label = nom if dia < 366 else None 
    plt.axvline(dia, color=color, linestyle='--', alpha=0.5, linewidth=1, label=label)

plt.ylabel("Altitud solar màxima (graus)")
plt.xlabel("Dia (d)")
plt.axvline(365.5, color='black', linestyle='--', linewidth=2, alpha=0.5)
plt.grid(True, alpha=0.3)
plt.xlim(1, n_dies)
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('altitud_solar_final.png', dpi=300)
plt.show()

# produccio diaria
plt.figure(figsize=(12, 5))
plt.plot(dies_plot, produccio_diaria, 'g-', linewidth=1)
plt.fill_between(dies_plot, produccio_diaria, alpha=0.2, color='green')

for dia, nom, color in events:
    label = nom if dia < 366 else None 
    plt.axvline(dia, color=color, linestyle='--', alpha=0.5, linewidth=1, label=label)

plt.scatter([dia_max+1], [energia_max], c='red', s=100, zorder=5, label=f'Max: dia {dia_max+1}')
plt.scatter([dia_min+1], [energia_min], c='blue', s=100, zorder=5, label=f'Min: dia {dia_min+1}')
plt.axvline(365.5, color='black', linestyle='--', linewidth=2, alpha=0.5)
plt.ylabel("Energia produida (kWh)")
plt.xlabel("Dia (d)")
#plt.title("Producció diaria d'energia solar")
plt.grid(True, alpha=0.3)
plt.legend()
plt.xlim(1, n_dies)
plt.ylim(bottom=0)
plt.tight_layout()
plt.savefig('produccio_diaria_final.png', dpi=300)
plt.show()

# produccio en un dia
def plot_dia_detall(dia, energia, hores, pot, irr, color_pot, color_irr, tipus):
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    limit_y_pot = (num_panells * potencia_max / 1000) * 1.1
    
    ax1.set_xlabel("Hora del dia (h)")
    ax1.set_ylabel("Potència generada (kW)", color=color_pot)
    if len(hores) > 0:
        ax1.plot(hores, pot/1000, color=color_pot, linewidth=2.5, label='Potència')
        ax1.fill_between(hores, pot/1000, alpha=0.3, color=color_pot)
    ax1.tick_params(axis='y', labelcolor=color_pot)
    
    ax1.set_ylim(0, limit_y_pot) 
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Irradiació solar (W/m²)", color=color_irr)
    if len(hores) > 0:
        ax2.plot(hores, irr, color=color_irr, linewidth=2, linestyle='--', label='Irradiació')
    ax2.tick_params(axis='y', labelcolor=color_irr)
    
    ax2.set_ylim(0, 1400)

    # TÍTOL AMB DIA I TIPUS
    print(f"Dia {dia + 1}: {tipus} producció ({energia:.2f} kWh)")
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f'potencia_dia_{tipus.lower()}_final.png', dpi=300)
    plt.show()


plot_dia_detall(dia_max, energia_max, hores_dia_max, pot_dia_max, irr_dia_max, 'red', 'orange', 'Maxima')
plot_dia_detall(dia_min, energia_min, hores_dia_min, pot_dia_min, irr_dia_min, 'blue', 'cyan', 'Minima')

# calcular produccio i irradiacio mensual
def calcular_mensual(produccio_diaria, orbita_theta_terra, normal_vec):
    mesos = 24  # 2 anys
    produccio_mensual = np.zeros(mesos)
    irradiacio_mensual = np.zeros(mesos) # Ara serà en kWh/m2/dia
    
    dies_per_mes = 365 // 12
    dt_h = 1.0 / 6.0 # El mateix pas de temps que fas servir a calcular_produccio_horaria (10 min)
    
    for mes in range(mesos):
        dia_inici = mes * dies_per_mes
        dia_fi = min((mes + 1) * dies_per_mes, len(produccio_diaria))
        
        # produccio mensual (suma total del mes)
        produccio_mensual[mes] = np.sum(produccio_diaria[dia_inici:dia_fi])
        
        # irradiacio: calculem la mitjana d'energia rebuda per dia (kWh/m2)
        irr_diaria_acumulada_total = 0
        
        for dia in range(dia_inici, dia_fi):
            _, _, irr, _, _ = calcular_produccio_horaria(orbita_theta_terra[dia], dia, normal_vec)
            
            if len(irr) > 0:
                # Integrem la irradiació (W/m2) respecte al temps per obtenir Wh/m2
                # Després dividim per 1000 per tenir kWh/m2
                energia_solar_dia = np.sum(irr) * dt_h / 1000.0
                irr_diaria_acumulada_total += energia_solar_dia
        
        # Fem la mitjana diària d'aquell mes
        if dia_fi > dia_inici:
            irradiacio_mensual[mes] = irr_diaria_acumulada_total 
    
    return produccio_mensual, irradiacio_mensual

produccio_mensual, irradiacio_mensual = calcular_mensual(produccio_diaria, orbita_theta_terra, normal)

# Dades noves proporcionades (12 mesos)
dades_pvgis = [52.26, 52.68, 64.81, 65.29, 71.04, 70.67, 73.94, 71.23, 61.77, 55.74, 47.49, 49.09]
# Les repetim per cobrir els 24 mesos de la gràfica
dades_pvgis_plot = np.tile(dades_pvgis, 2)

# Calculem la diferència (Simulació - PVGIS)
diferencia = (produccio_mensual - dades_pvgis_plot)

# GRÀFIC COMPARATIU MENSUAL 
fig, ax1 = plt.subplots(figsize=(14, 6))

mesos_plot = np.arange(1, 25)
width = 0.35 # Amplada de les barres

ax1.set_xlabel("Mes")
ax1.set_ylabel("Energia produïda (kWh)", color='black')

# 1. Barres de la simulació (esquerra)
rects1 = ax1.bar(mesos_plot - width/2, produccio_mensual, width, alpha=0.6, color='green', label='Mensual simulada')
# 2. Barres de les dades PVGIS (dreta)
rects2 = ax1.bar(mesos_plot + width/2, dades_pvgis_plot, width, alpha=0.6, color='purple', label='PVGIS')


# 3. Línia de Diferència
line_diff = ax1.plot(mesos_plot, diferencia, color='blue', linestyle='--', linewidth=2, marker='x', label='Diferència (Sim - PVGIS)')
# Omplim l'àrea sota la diferència per visualitzar millor la desviació
ax1.fill_between(mesos_plot, 0, diferencia, color='blue', alpha=0.1)
# Marquem la línia 0 per referència clara
ax1.axhline(0, color='black', linewidth=1, linestyle='-', alpha=0.3)

ax1.tick_params(axis='y', labelcolor='black')
ax1.grid(True, alpha=0.3, axis='y')

# EIX SECUNDARI (irradiació) 
ax2 = ax1.twinx()
ax2.set_ylabel("Irradiació mensual (kWh/m²)", color='orange')
line_irr = ax2.plot(mesos_plot, irradiacio_mensual, color='orange', linewidth=2.5, 
                 marker='o', markersize=6, label='Irradiació mensual')
ax2.tick_params(axis='y', labelcolor='orange')
ax2.set_ylim(0,350)

ax1.axvline(12.5, color='black', linestyle='--', linewidth=2, alpha=0.5)

# LLEGENDA COMBINADA 
# Hem d'ajuntar els handles de l'eix 1 (barres i línia diff) i l'eix 2 (irradiació)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right', framealpha=0.9)

plt.tight_layout()
plt.savefig('produccio_comparativa_completa.png', dpi=300)
plt.show()


# CÀLCUL AMB IRRADIACIÓ DONADA

# Dades proporcionades
irradiacio_input = [4.36, 4.94, 5.6, 5.96, 6.39, 6.71, 6.87, 6.59, 5.79, 4.9, 4.19, 4.1]
dades_pvgis = [52.26, 52.68, 64.81, 65.29, 71.04, 70.67, 73.94, 71.23, 61.77, 55.74, 47.49, 49.09]

produccio_calculada_model = []

for m in range(12):
    # Irradiació diària mitjana (kWh/m2)
    H = irradiacio_input[m]
    
    # Estimació temperatura ambient mitjana (promig max/min mes)
    t_avg = (temps_max_mes[m] + temps_min_mes[m]) / 2.0
    
    # Estimació temperatura panell
    # Suposem una irradiància mitjana durant les hores de sol per calcular l'escalfament
    # (aprox 12h de mitjana anual per simplificar càlcul tèrmic)
    g_avg = (H * 1000.0) / 12.0 
    t_panel = t_avg + 0.025 * g_avg
    
    # Factor de pèrdua tèrmica (model)
    perdua = np.maximum(0, t_panel - 25.0) * coef_temp
    factor_eff = 1.0 - perdua
    
    # Producció ideal diària = H (kWh/m2) * Potència (kW) * Panells
    # (assumint 1kW/m2 standard test condition)
    prod_ideal_diaria = H * (potencia_max * num_panells / 1000.0)
    
    # Producció real diària amb factor tèrmic
    prod_real_diaria = prod_ideal_diaria * factor_eff
    
    # Producció mensual (dies aprox per mes 30.44)
    dies_mes = 365.25 / 12.0
    prod_mensual = prod_real_diaria * dies_mes
    
    produccio_calculada_model.append(prod_mensual)


# GRÀFIC COMPARATIU amb irradiació del PVGIS
fig, ax = plt.subplots(figsize=(12, 6))

mesos_idx = np.arange(1, 13)
width = 0.35

ax.bar(mesos_idx - width/2, produccio_calculada_model, width, label='Model (amb Irradiació PVGIS)', color='teal', alpha=0.7)
ax.bar(mesos_idx + width/2, dades_pvgis, width, label='PVGIS', color='purple', alpha=0.7)

ax.set_ylabel('Energia Mensual (kWh)')
ax.set_xlabel('Mes')
ax.set_xticks(mesos_idx)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('comparativa_irradiacio_donada.png', dpi=300)
plt.show()
