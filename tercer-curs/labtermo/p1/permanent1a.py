import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

fitxer_csv = "dadespermanent_1a.csv"

# llegim les dades
df = pd.read_csv(fitxer_csv, header=None)

# eliminem la primera fila (dada incorrecta)
df = df.iloc[1:]

# convertim les columnes a float
df = df.astype(float)

temps = df[0].values

# noms de les columnes (1 a 6)
noms_columnes = [
    "Barra gran (0 cm)",
    "Barra gran (10 cm)",
    "Barra gran (15 cm)",
    "Barra petita (0 cm)",
    "Barra petita (10 cm)",
    "Barra petita (20 cm)"
]

resultats = {}

for i in range(1, 7):
    y = df[i].values
    
    # temperatura mitjana
    temp_mitjana = np.mean(y)
    
    # amplitud (max - min) / 2
    amplitud = (np.max(y) - np.min(y)) / 2
    
    # periode calcular distancia entre pics consecutius
    pics, _ = find_peaks(y)
    if len(pics) > 1:
        periodes = np.diff(temps[pics])
        periode_mitja = np.mean(periodes)
    else:
        periode_mitja = np.nan
    
    resultats[noms_columnes[i-1]] = {
        "Temperatura mitjana (°C)": temp_mitjana,
        "Amplitud (°C)": amplitud,
        "Periode (s)": periode_mitja
    }
    
    plt.figure()
    plt.plot(temps, y, marker='o', markersize=3)
    plt.xlabel("Temps (s)")
    plt.ylabel(f"Temperatura {noms_columnes[i-1]} (°C)")
    plt.title(f"Temperatura de {noms_columnes[i-1]} en el temps")
    plt.grid(True)
    plt.savefig(f"graf_{i}_{noms_columnes[i-1].replace(' ', '_').replace('(', '').replace(')', '')}.png")
    plt.close()

plt.figure()
for i in range(1, 7):
    plt.plot(temps, df[i].values, marker='o', markersize=3, label=noms_columnes[i-1])
plt.xlabel("Temps (s)")
plt.ylabel("Temperatura (°C)")
plt.title("Totes les posicions en el temps")
plt.legend()
plt.grid(True)
plt.savefig("graf_totes_posicions.png")
plt.show()

# calcul del desfasament aproximat
for j in [1, 2]:
    pics_ref, _ = find_peaks(df[1].values)
    pics_col, _ = find_peaks(df[j+1].values)
    if len(pics_ref) > 0 and len(pics_col) > 0:
        desfasament = temps[pics_col[0]] - temps[pics_ref[0]]
        resultats[noms_columnes[j]]["Desfasament respecte " + noms_columnes[0]] = desfasament

for j in [4, 5]:
    pics_ref, _ = find_peaks(df[4].values)
    pics_col, _ = find_peaks(df[j].values)
    if len(pics_ref) > 0 and len(pics_col) > 0:
        desfasament = temps[pics_col[0]] - temps[pics_ref[0]]
        resultats[noms_columnes[j]]["Desfasament respecte " + noms_columnes[3]] = desfasament

for columna, dades in resultats.items():
    print(f"\n{columna}:")
    for key, value in dades.items():
        print(f"  {key}: {value:.3f}")

fitxer_resultats = "resultats_temperatures.txt"

with open(fitxer_resultats, "w") as f:
    for columna, dades in resultats.items():
        f.write(f"{columna}:\n")
        print(f"{columna}:")  # imprimim per pantalla
        for key, value in dades.items():
            f.write(f"  {key}: {value:.3f}\n")
            print(f"  {key}: {value:.3f}")
        f.write("\n")
        print()