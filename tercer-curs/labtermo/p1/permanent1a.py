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

# funcio per calcular periode via FFT
# fem servir la FFT per obtenir la frequencia dominant de la senyal
# el metode de find_peaks podia donar periodes erronis si hi havia soroll o pics irregulars
def calcular_periode_fft(y, dt):
    y = y - np.mean(y)  # treure mitjana per eliminar component DC
    N = len(y)
    fft_y = np.fft.fft(y)  # calcul FFT de la senyal
    freqs = np.fft.fftfreq(N, dt)  # corresponent vector de frequencies
    amplitud = np.abs(fft_y[:N//2])  # magnitud positiva
    freqs_pos = freqs[:N//2]

    # ignorem freq = 0
    amplitud[0] = 0

    # trobem frequencia dominant
    idx_max = np.argmax(amplitud)
    freq_dom = freqs_pos[idx_max]

    if freq_dom == 0:
        return np.nan
    else:
        periode = 1 / freq_dom # com que la senyal es sinusoidal, podem assumir que T=1/f_dom
        return periode

# calculs principals per cada columna
for i in range(1, 7):
    y = df[i].values
    temp_mitjana = np.mean(y)
    amplitud = (np.max(y) - np.min(y)) / 2 # encara que no es la amplitud exacta es acceptable
    # com hem dit abans findpeaks no era fiable

    # calcul periode via FFT
    periode_mitja = calcular_periode_fft(y, dt)
    sigma_periode = dt  # aproximem incertesa del periode a dt

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

# desfasament respecte a posicio inicial de cada barra
parells_desfasament = [
    (1, "Barra gran (0 cm)", 2, "Barra gran (10 cm)"),
    (1, "Barra gran (0 cm)", 3, "Barra gran (15 cm)"),
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
    sigma_desfasament = dt  # incertesa del desfasament
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
