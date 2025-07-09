# MODELO HÍBRIDO SIR + ABM
# Combina un modelo SIR (ecuaciones diferenciales) con un modelo ABM (basado en agentes)
# El modelo SIR inicializa el ABM, y luego el ABM ajusta dinámicamente el beta usado en SIR

import random
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
from ModeloSIR import modelo_SRI

# -----------------------------
# MODELO ABM
# -----------------------------

class Person:
    #Inicializacion de persona
    def __init__(self, id, age, mask_usage):
        self.id = id
        self.age = age
        self.state = "S"
        self.mask_usage = mask_usage
        self.super_spreader = random.random() < 0.05  # Probabilidad de ser alguien que pasa la enfermedad a más personas

    #Funcion de interaccion entre personas
    def interact(self, others, base_infection_rate):
        contacts = random.randint(10, 20) if self.super_spreader else random.randint(1, 5)
        for _ in range(contacts):
            other = random.choice(others)
            # Caso de que una persona suceptible y una persona infectada interactuen
            if (self.state == "S" and other.state == "I") or (self.state == "I" and other.state == "S"):
                prob = base_infection_rate * (1 - self.mask_usage) * (1 - other.mask_usage)
                if random.random() < prob:
                    if self.state == "S":
                        self.state = "I"
                    else:
                        other.state = "I"

    # Intentar recuperarse de la enfermedad
    def try_to_recover(self, recovery_rate):
        if self.state == "I" and random.random() < recovery_rate:
            self.state = "R"

# Simulación del ABM por un número fijo de días
# Se inicializa con cantidades específicas de S, I, R

def simulateABM_with_initial_state(S0, I0, R0, base_infection_rate, recovery_rate, dias, mask_compliance):
    population = []

    for i in range(S0):
        mask_usage = 0.9 if random.random() < mask_compliance else 0.1
            
        p = Person(i, random.randint(10, 80), mask_usage)
        p.state = "S"
        population.append(p)
    for i in range(I0):
        mask_usage = 0.9 if random.random() < mask_compliance else 0.1
        
        p = Person(S0 + i, random.randint(10, 80), mask_usage)
        p.state = "I"
        population.append(p)
    for i in range(R0):
        mask_usage = 0.9 if random.random() < mask_compliance else 0.1
        
        p = Person(S0 + I0 + i, random.randint(10, 80), mask_usage)
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
# MODELO HÍBRIDO
# -----------------------------

def run_hybrid_model(S0, I0, R0, beta_inicial, gamma, dias, abm_window=1, vacunacion_diaria=0, cumplimiento_mascarilla=0):
    poblacion_total = S0 + I0 + R0
    y = [S0, I0, R0]
    resultados_sir = [y]
    betas = [beta_inicial]

    for dia in range(dias):
        # Simular el ABM por un paso (p.ej., 1 día)
        abm_result = simulateABM_with_initial_state(int(y[0]), int(y[1]), int(y[2]), beta_inicial, gamma, abm_window, cumplimiento_mascarilla)
        S_abm, I_abm, R_abm, nuevos_contagios = abm_result[-1]
        # Cálculo diario de vacunaciones
        vacunados = vacunacion_diaria * S_abm
        S_abm -= vacunados
        R_abm += vacunados

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
# RESULTADOS
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

def imprimir_dias_pico(resultados):
    S = resultados[:, 0]
    I = resultados[:, 1]
    R = resultados[:, 2]

    dia_pico_S = int(np.argmax(S))
    dia_pico_I = int(np.argmax(I))
    dia_pico_R = int(np.argmax(R))

    print("\nDías pico por clase:")
    print(f"Susceptibles: día {dia_pico_S} con {int(S[dia_pico_S])} personas")
    print(f"Infectados: día {dia_pico_I} con {int(I[dia_pico_I])} personas")
    print(f"Recuperados: día {dia_pico_R} con {int(R[dia_pico_R])} personas")

# -----------------------------
# EJECUCIÓN PRINCIPAL
# -----------------------------

if __name__ == "__main__":
    # Condiciones iniciales
    S0, I0, R0 = 990, 10, 0
    beta0 = 0.4
    gamma = 0.1
    dias = 100

    porcentaje_vacunacion = 0.0
    porcentaje_mascarilla = 0.8

    resultados, betas = run_hybrid_model(S0, I0, R0, beta0, gamma, dias, 1, porcentaje_vacunacion, porcentaje_mascarilla)
    plot_hybrid(resultados, betas)
    imprimir_dias_pico(resultados)