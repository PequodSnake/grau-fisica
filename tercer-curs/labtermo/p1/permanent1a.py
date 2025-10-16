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
        periode = 1 / freq_dom
        return periode

# calculs principals per cada columna
for i in range(1, 7):
    y = df[i].values
    temp_mitjana = np.mean(y)
    amplitud = (np.max(y) - np.min(y)) / 2

    # calcul periode via FFT
    periode_mitja = calcular_periode_fft(y, dt)
    sigma_periode = dt

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
    # trobar pics de referencia i columna
    pics_ref, _ = find_peaks(df[ref_idx].values)
    pics_col, _ = find_peaks(df[col_idx].values)

    # temps dels pics
    t_pics_ref = temps[pics_ref]
    t_pics_col = temps[pics_col]

    # triem el primer pic de la referencia
    t_ref = t_pics_ref[0]

    # busquem el primer pic de la columna que vingui despres
    t_col_candidates = t_pics_col[t_pics_col > t_ref]
    if len(t_col_candidates) == 0:
        # si no n'hi ha cap posterior, agafem el primer
        t_col = t_pics_col[0]
    else:
        t_col = t_col_candidates[0]

    # desfasament temporal
    desfasament_s = t_col - t_ref

    # si el desfasament supera mig periode, ajustem
    periode = resultats[col_nom]["Periode (s)"]
    if desfasament_s > periode / 2:
        desfasament_s -= periode

    # desfasament angular
    desfasament_phi = 2 * np.pi * desfasament_s / periode
    sigma_periode = resultats[col_nom]["sigma_P (s)"]
    sigma_desfasament = dt
    sigma_phi = desfasament_phi * np.sqrt((sigma_desfasament/desfasament_s)**2 + (sigma_periode/periode)**2)

    # guardem resultats
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
