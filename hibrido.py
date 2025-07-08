# MODELO HÍBRIDO SIR + ABM
# Combina un modelo SIR (ecuaciones diferenciales) con un modelo ABM (basado en agentes)
# El modelo SIR inicializa el ABM, y luego el ABM ajusta dinámicamente el beta usado en SIR

import random
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
from agente import Person

# -----------------------------
# MODELO ABM
# -----------------------------

# Simulación del ABM por un número fijo de días
# Se inicializa con cantidades específicas de S, I, R

def simulateABM_with_initial_state(S0, I0, R0, base_infection_rate, recovery_rate, dias):
    population = []

    for i in range(S0):
        p = Person(i, random.randint(10, 80))
        p.state = "S"
        population.append(p)
    for i in range(I0):
        p = Person(S0 + i, random.randint(10, 80))
        p.state = "I"
        population.append(p)
    for i in range(R0):
        p = Person(S0 + I0 + i, random.randint(10, 80))
        p.state = "R"
        population.append(p)

    history = []
    for day in range(dias):
        prev_infected = sum(1 for p in population if p.state == "I")

        for person in population:
            person.interact(population, base_infection_rate)
        for person in population:
            person.try_to_recover(recovery_rate)

        counts = {"S": 0, "I": 0, "R": 0}
        for person in population:
            counts[person.state] += 1

        new_infections = counts["I"] - prev_infected
        new_infections = max(new_infections, 0)

        history.append((counts["S"], counts["I"], counts["R"], new_infections))

    return history

# -----------------------------
# MODELO SIR
# -----------------------------

def modelo_SRI(y, t, beta, gamma):
    S, I, R = y
    N = S + I + R
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

# -----------------------------
# MODELO HÍBRIDO
# -----------------------------

def run_hybrid_model(S0, I0, R0, beta_inicial, gamma, dias, abm_window=1):
    poblacion_total = S0 + I0 + R0
    y = [S0, I0, R0]
    resultados_sir = [y]
    betas = [beta_inicial]

    for dia in range(dias):
        # Simular el ABM por un paso (p.ej., 1 día)
        abm_result = simulateABM_with_initial_state(int(y[0]), int(y[1]), int(y[2]), beta_inicial, gamma, abm_window)
        S_abm, I_abm, R_abm, nuevos_contagios = abm_result[-1]

        # Calcular nuevo beta desde el ABM
        if S_abm > 0 and I_abm > 0:
            beta_est = (nuevos_contagios * poblacion_total) / (S_abm * I_abm)
        else:
            beta_est = 0

        betas.append(beta_est)

        # Simular el modelo SIR por un paso con el nuevo beta
        t_range = np.linspace(0, 1, 2)
        y_next = odeint(modelo_SRI, y, t_range, args=(beta_est, gamma))
        y = y_next[-1]
        resultados_sir.append(y.tolist())

    return np.array(resultados_sir), betas

# -----------------------------
# GRAFICACIÓN
# -----------------------------

def plot_hybrid(resultados, betas):
    S, I, R = resultados[:, 0], resultados[:, 1], resultados[:, 2]
    dias = range(len(S))

    plt.figure(figsize=(12, 6))

    # Graficar poblaciones
    plt.subplot(1, 2, 1)
    plt.plot(dias, S, label="Susceptibles")
    plt.plot(dias, I, label="Infectados")
    plt.plot(dias, R, label="Recuperados")
    plt.title("Modelo Híbrido SIR + ABM")
    plt.xlabel("Días")
    plt.ylabel("Personas")
    plt.legend()
    plt.grid(True)

    # Graficar beta estimado
    plt.subplot(1, 2, 2)
    plt.plot(dias, betas, label="Beta estimado", color="orange")
    plt.title("Evolución de Beta desde ABM")
    plt.xlabel("Días")
    plt.ylabel("Beta")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# -----------------------------
# EJECUCIÓN PRINCIPAL
# -----------------------------

if __name__ == "__main__":
    # Condiciones iniciales
    S0, I0, R0 = 990, 10, 0
    beta0 = 0.3
    gamma = 0.1
    dias = 50

    resultados, betas = run_hybrid_model(S0, I0, R0, beta0, gamma, dias)
    plot_hybrid(resultados, betas)
