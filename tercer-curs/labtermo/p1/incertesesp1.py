import math
import numpy as np

def calc_uK1(p1, p2, K2, uK2, up1, up2):
    term1 = (p1**4 / p2**4) * uK2**2
    term2 = 4 * (p2**2 / p1**4) * (K2**2) * (up2**2)
    term3 = 4 * (p2**4 / p1**6) * (K2**2) * (up1**2)
    
    uK1 = math.sqrt(term1 + term2 + term3)
    return uK1

p1 = 3.85
p2 = 6.61
K2 = 80.4
uK2 = 0
up1 = 0.037
up2 = 0.26

uK1 = calc_uK1(p1, p2, K2, uK2, up1, up2)
print(f"u_K1 = {uK1:.6f}")

def calc_m(ai, aj, xi, xj, dai, daj):
    m = np.log(ai / aj) / (xj - xi)
    dm = np.sqrt(
        (1 / ((xj - xi) * ai) * dai)**2 +
        (1 / ((xj - xi) * aj) * daj)**2
    )
    return m, dm

def calc_h(dphi, ddphi, dx):
    h = dphi / dx
    dh = ddphi / dx
    return h, dh

def calc_lambda(kappa, r, m, dm, h, dh):
    lam = kappa * r * (m**2 - h**2) / 2
    dlam = np.sqrt((kappa * r * m * dm)**2 + (kappa * r * h * dh)**2)
    return lam, dlam

# Barra petita
ai_s, aj_s = 6.0, 2.7
xi_s, xj_s = 0, 0.1
dai_s, daj_s = 0.1, 0.1

dphi_s, ddphi_s = 0.243, 0.084
dx_s = 0.1
r_s = 0.015

# Barra gran
ai_g, aj_g = 3.5, 1.6
xi_g, xj_g = 0, 0.1
dai_g, daj_g = 0.1, 0.1

dphi_g, ddphi_g = 0.324, 0.083
dx_g = 0.05
r_g = 0.0255

kappa = 205 

# Barra petita
m_s, dm_s = calc_m(ai_s, aj_s, xi_s, xj_s, dai_s, daj_s)
h_s, dh_s = calc_h(dphi_s, ddphi_s, dx_s)
lam_s, dlam_s = calc_lambda(kappa, r_s, m_s, dm_s, h_s, dh_s)

# Barra gran
m_g, dm_g = calc_m(ai_g, aj_g, xi_g, xj_g, dai_g, daj_g)
h_g, dh_g = calc_h(dphi_g, ddphi_g, dx_g)
lam_g, dlam_g = calc_lambda(kappa, r_g, m_g, dm_g, h_g, dh_g)

print("Barra petita")
print(f"m = {m_s:.5f} ± {dm_s:.5f}")
print(f"h = {h_s:.5f} ± {dh_s:.5f}")
print(f"λ = {lam_s:.5e} ± {dlam_s:.5e}")

print("Barra gran")
print(f"m = {m_g:.5f} ± {dm_g:.5f}")
print(f"h = {h_g:.5f} ± {dh_g:.5f}")
print(f"λ = {lam_g:.5e} ± {dlam_g:.5e}")