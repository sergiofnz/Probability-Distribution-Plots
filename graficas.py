import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Crear subplots
fig, axs = plt.subplots(3, 1, figsize=(8, 12))

# 1. Distribución de Poisson - Variando λ
lambdas = [1, 4, 10]
x_poisson = np.arange(0, 20)
for lam in lambdas:
    poisson_dist = stats.poisson.pmf(x_poisson, lam)
    axs[0].plot(x_poisson, poisson_dist, label=f"λ={lam}")
axs[0].set_title("Distribución de Poisson")
axs[0].set_xlabel("Número de eventos")
axs[0].set_ylabel("Probabilidad")
axs[0].legend(loc="upper right")

# 2. Distribución Exponencial - Variando λ
lambdas_exp = [0.5, 1, 2]
x_exp = np.linspace(0, 5, 100)
for lam in lambdas_exp:
    exp_dist = stats.expon.pdf(x_exp, scale=1/lam)
    axs[1].plot(x_exp, exp_dist, label=f"λ={lam}")
axs[1].set_title("Distribución Exponencial")
axs[1].set_xlabel("Tiempo entre eventos")
axs[1].set_ylabel("Densidad de Probabilidad")
axs[1].legend(loc="upper right")

# 3. Distribución Normal - Variando σ (media fija en 0)
means = [0]
std_devs = [0.5, 1, 2]
x_norm = np.linspace(-5, 5, 100)
for sigma in std_devs:
    norm_dist = stats.norm.pdf(x_norm, loc=0, scale=sigma)
    axs[2].plot(x_norm, norm_dist, label=f"σ={sigma}")
axs[2].set_title("Distribución Normal")
axs[2].set_xlabel("Valor")
axs[2].set_ylabel("Densidad de Probabilidad")
axs[2].legend(loc="upper right")

plt.tight_layout()
plt.show()
