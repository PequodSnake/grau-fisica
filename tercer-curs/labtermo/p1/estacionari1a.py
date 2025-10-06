import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

FITXER = 'dadesestacionari_1a.csv'
OUT_DIR = 'fit_results'
os.makedirs(OUT_DIR, exist_ok=True)

# funcio model exponencial
def T_model(x, T0, a, b):
    return T0 + a * np.exp(b * x)

# llegim dades
df = pd.read_csv(FITXER, sep=';', decimal=',', header=0)
metalls = {
    'Ferro': (df['x_fe'], df['T_fe']),
    'Llauto': (df['x_ll'], df['T_ll']),
    'Alumini': (df['x_al'], df['T_al'])
}

resultats = []
plt.figure(figsize=(8,6))

for nom, (x, T) in metalls.items():
    x, T = np.array(x), np.array(T)
    T0_ini, a_ini, b_ini = T[-1], T[0]-T[-1], -0.03  # valors inicials per al fit

    # ajust exponencial
    popt, pcov = curve_fit(T_model, x, T, p0=[T0_ini, a_ini, b_ini],
                           bounds=([0,0,-1],[300,300,0]), maxfev=20000)
    T0, a, b = popt
    T0_err, a_err, b_err = np.sqrt(np.diag(pcov))  # incerteses del fit

    # regressio lineal sobre ln(T-T0)
    mask = (T - T0) > 0  # només valors positius per log
    m, n = np.polyfit(x[mask], np.log(T[mask]-T0), 1)
    # error de m
    residus = np.log(T[mask]-T0) - (m*x[mask] + n)
    sigma_m = np.sqrt(np.sum(residus**2)/(len(residus)-2) / np.sum((x[mask]-np.mean(x[mask]))**2))
    R2_lin = np.corrcoef(np.log(T[mask]-T0), m*x[mask]+n)[0,1]**2

    # R^2 del fit
    T_fit = T_model(x, T0, a, b)
    R2 = 1 - np.sum((T-T_fit)**2)/np.sum((T-np.mean(T))**2)

    # p i error de p a partir de m
    p, p_err = -m*100, sigma_m*100

    resultats.append((nom, T0, T0_err, a, a_err, b, b_err, R2, p, p_err, n, m, R2_lin))

    # grafic dades i fit
    x_fit = np.linspace(min(x), max(x), 400)
    plt.scatter(x, T, label=f'dades experimentals ({nom})')
    plt.plot(x_fit, T_model(x_fit, T0, a, b), label=f'regressió exp ({nom})')

plt.xlabel('Distància (cm)')
plt.ylabel('Temperatura (°C)')
plt.title('Ajust exponencial: T(x)=T0+a exp(bx)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'ajust_exponencial.png'), dpi=300)
plt.close()

# grafic ln(T-T0)
plt.figure(figsize=(8,6))
for nom, T0, *_ , n, m, _ in resultats:
    x, T = metalls[nom]
    mask = (T - T0) > 0
    plt.scatter(x[mask], np.log(T[mask]-T0), label=f'dades experimentals ({nom})')
    x_fit = np.linspace(min(x), max(x), 400)
    plt.plot(x_fit, n + m*x_fit, label=f'regressió lin ({nom})')
plt.xlabel('Distància (cm)')
plt.ylabel('ln(T-T0)')
plt.legend(); plt.grid(True)
plt.title('Regressió lineal de ln(T-T0)')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'regressio_lnT.png'), dpi=300)
plt.close()

# escrivim resultats a TXT
with open(os.path.join(OUT_DIR, 'resultats.txt'), 'w', encoding='utf-8') as f:
    for nom, T0, T0_err, a, a_err, b, b_err, R2, p, p_err, n, m, R2_lin in resultats:
        f.write(f'{nom}:\n')
        f.write(f'  T0 = {T0:.2f} ± {T0_err:.2f} °C\n')
        f.write(f'  a  = {a:.2f} ± {a_err:.2f}\n')
        f.write(f'  b  = {b:.5f} ± {b_err:.5f} cm^-1\n')
        f.write(f'  R^2_exp = {R2:.5f}\n')
        f.write(f'  p  = {p:.5f} ± {p_err:.5f} m^-1\n')
        f.write(f'  n  = {n:.3f},  m = {m:.5f} cm^-1,  R^2_lin = {R2_lin:.5f}\n\n')

print("resultats guardats a fit_results")

