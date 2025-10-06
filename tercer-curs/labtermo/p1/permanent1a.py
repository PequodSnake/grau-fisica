import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

fitxer_csv = "dadespermanent_1a.csv"

# llegim dades i convertim a float
df = pd.read_csv(fitxer_csv, header=None)
df = df.iloc[1:].astype(float)
temps = df[0].values
dt = temps[1] - temps[0]  # resolucio temporal

# noms columnes
noms_columnes = [
    "Barra gran (0 cm)",
    "Barra gran (10 cm)",
    "Barra gran (15 cm)",
    "Barra petita (0 cm)",
    "Barra petita (10 cm)",
    "Barra petita (20 cm)"
]

# incerteses
sigma_T = 0.1
sigma_x = 0.1

resultats = {}

# calculs principals per cada columna
for i in range(1, 7):
    y = df[i].values
    temp_mitjana = np.mean(y)
    amplitud = (np.max(y) - np.min(y)) / 2

    pics, _ = find_peaks(y)  # trobem els pics per calcular periode

    if len(pics) > 1:
        periodes = np.diff(temps[pics])  # diferencies entre pics consecutius
        periode_mitja = np.mean(periodes)
        sigma_periode = np.std(periodes, ddof=1) / np.sqrt(len(periodes))  # incertesa periode
    else:
        periode_mitja = np.nan
        sigma_periode = np.nan

    resultats[noms_columnes[i-1]] = {
        "Temperatura mitjana (C)": temp_mitjana,
        "Amplitud (C)": amplitud,
        "sigma_T (C)": sigma_T,
        "Periode (s)": periode_mitja,
        "sigma_P (s)": sigma_periode
    }

    # grafic individual
    plt.figure()
    plt.plot(temps, y, 'o-', markersize=3)
    plt.xlabel("Temps (s)")
    plt.ylabel("Temperatura (C)")
    plt.title(noms_columnes[i-1])
    plt.grid(True)
    plt.savefig(f"graf_{i}_{noms_columnes[i-1].replace(' ', '_').replace('(', '').replace(')', '')}.png")
    plt.close()

# grafic de totes les posicions
plt.figure()
for i in range(1, 7):
    plt.plot(temps, df[i].values, 'o-', markersize=3, label=noms_columnes[i-1])
plt.xlabel("Temps (s)")
plt.ylabel("Temperatura (C)")
plt.title("Totes les posicions en el temps")
plt.legend()
plt.grid(True)
plt.savefig("graf_totes_posicions.png")
plt.show()

# desfasament
parells_desfasament = [
    # barres grans respecte 0 cm
    (1, "Barra gran (0 cm)", 2, "Barra gran (10 cm)"),
    (1, "Barra gran (0 cm)", 3, "Barra gran (15 cm)"),
    # barres petites respecte 0 cm
    (4, "Barra petita (0 cm)", 5, "Barra petita (10 cm)"),
    (4, "Barra petita (0 cm)", 6, "Barra petita (20 cm)")
]

for ref_idx, ref_nom, col_idx, col_nom in parells_desfasament:
    pics_ref, _ = find_peaks(df[ref_idx].values)  # pics de la columna referencia
    pics_col, _ = find_peaks(df[col_idx].values)  # pics de la columna a comparar

    desfasament_s = temps[pics_col[0]] - temps[pics_ref[0]]  # desfasament en segons
    periode = resultats[col_nom]["Periode (s)"]
    sigma_periode = resultats[col_nom]["sigma_P (s)"]
    desfasament_phi = 2 * np.pi * desfasament_s / periode  # desfasament en radians
    sigma_desfasament = dt  # incertesa desfasament
    # incertesa del desfasament en radians combinant errors
    sigma_phi = desfasament_phi * np.sqrt((sigma_desfasament/desfasament_s)**2 + (sigma_periode/periode)**2)

    # guardem els resultats del desfasament
    resultats[col_nom][f"Desfasament respecte {ref_nom} (s)"] = desfasament_s
    resultats[col_nom]["sigma_desfasament (s)"] = sigma_desfasament
    resultats[col_nom][f"Desfasament respecte {ref_nom} (phi rad)"] = desfasament_phi
    resultats[col_nom]["sigma_phi (rad)"] = sigma_phi

# escrivim resultats a TXT
with open("resultats_temperatures.txt", "w", encoding="utf-8") as f:
    for columna, dades in resultats.items():
        f.write(f"{columna}:\n")
        print(f"\n{columna}:")
        for key, value in dades.items():
            f.write(f"  {key}: {value:.4f}\n")
            print(f"  {key}: {value:.4f}")
        f.write("\n")
        print()
