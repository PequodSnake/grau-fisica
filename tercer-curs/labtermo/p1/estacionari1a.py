import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import linregress
import os

FITXER_DADES = 'dadesestacionari_1a.csv'
DIR_SORTIDA = 'fit_results'

# funcio exponencial per ajustar les dades
def exponencial(x, a, b):
    return a * np.exp(b * x)

# llegim el csv amb separador ';' i coma decimal
df = pd.read_csv(FITXER_DADES, sep=';', decimal=',', header=0)

# extraccio de dades
x_ferro, T_ferro = df['x_fe'].to_numpy(float), df['T_fe'].to_numpy(float)
x_llauto, T_llauto = df['x_ll'].to_numpy(float), df['T_ll'].to_numpy(float)
x_alumini, T_alumini = df['x_al'].to_numpy(float), df['T_al'].to_numpy(float)

# dades de cada metall
metalls = {
    'Ferro': (x_ferro, T_ferro),
    'Llauto': (x_llauto, T_llauto),
    'Alumini': (x_alumini, T_alumini)
}

resultats_exp = []
plt.figure(figsize=(8, 6))

# ajust exponencial
for nom, (x, T) in metalls.items():
    par_opt, par_cov = curve_fit(exponencial, x, T, p0=[T[0], -0.01])
    a, b = par_opt
    a_err, b_err = np.sqrt(np.diag(par_cov))

    T_ajust = exponencial(x, a, b)
    residus = T - T_ajust
    ss_res = np.sum(residus**2)
    ss_tot = np.sum((T - np.mean(T))**2)
    r2_exp = 1 - (ss_res / ss_tot)

    resultats_exp.append((nom, a, a_err, b, b_err, r2_exp))

    plt.scatter(x, T, label=f'{nom} dades')
    plt.plot(x, T_ajust, label=f'{nom} ajust exp')

plt.xlabel('Distancia (cm)')
plt.ylabel('Temperatura (°C)')
plt.title('Ajust exponencial de la temperatura')
plt.legend()
os.makedirs(DIR_SORTIDA, exist_ok=True)
plt.savefig(os.path.join(DIR_SORTIDA, 'ajust_exponencial.png'), dpi=300)
plt.close()

plt.figure(figsize=(8, 6))

# regressio lineal del ln(T)
resultats_lin = []
for nom, (x, T) in metalls.items():
    lnT = np.log(T)
    pendent, interseccio, r_valor, p_valor, err_std = linregress(x, lnT)
    r2_lin = r_valor**2

    lnT_ajust = interseccio + pendent * x

    plt.scatter(x, lnT, label=f'{nom} dades')
    plt.plot(x, lnT_ajust, label=f'{nom} ajust lin.')

    resultats_lin.append((nom, interseccio, pendent, err_std, r2_lin))

plt.xlabel('Distancia (cm)')
plt.ylabel('ln(T)')
plt.title('Regressio lineal de ln(T) per a totes les barres')
plt.legend()
plt.savefig(os.path.join(DIR_SORTIDA, 'regressio_lnT.png'), dpi=300)
plt.close()

# escrivim els resultats a un fitxer txt
with open(os.path.join(DIR_SORTIDA, 'resultats.txt'), 'w', encoding='utf-8') as f:
    f.write('Ajust exponencial T = a·exp(bx)\n')
    for nom, a, a_err, b, b_err, r2 in resultats_exp:
        f.write(f'{nom}: a = {a:.3f} ± {a_err:.3f}, b = {b:.5f} ± {b_err:.5f}, R² = {r2:.5f}\n')

    f.write('\nRegressio lineal de ln(T) = m·x + n\n')
    for nom, n, m, err, r2 in resultats_lin:
        f.write(f'{nom}: n = {n:.3f}, m = {m:.5f} ± {err:.5f}, R² = {r2:.5f}\n')

print('Resultats i grafiques guardats a la carpeta fit_results')
