import random
import matplotlib.pyplot as plt

class Person:
    #Inicializacion de persona
    def __init__(self, id, age):
        self.id = id
        self.age = age
        self.state = "S"
        self.mask_usage = random.uniform(0, 1)
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

# Simulacion ABM
def simulateABM(num_agents, days, base_infection_rate, recovery_rate):
    population = [Person(i, random.randint(10, 80)) for i in range(num_agents)]
    patient_zero = random.choice(population)
    patient_zero.state = "I"

    #Historial para graficación
    count = []

    #Maximos
    max_counts = {
        "S": {"day": 0, "value": 0},
        "I": {"day": 0, "value": 0},
        "R": {"day": 0, "value": 0}
    }

    # Simulacion por dias
    for day in range(days):
        for person in population:
            person.interact(population, base_infection_rate)
        for person in population:
            person.try_to_recover(recovery_rate)

        counts = {"S": 0, "I": 0, "R": 0}
        for person in population:
            counts[person.state] += 1

        for state in counts:
            if counts[state] > max_counts[state]["value"]:
                max_counts[state]["value"] = counts[state]
                max_counts[state]["day"] = day

        count.append({
            "counts": counts,
        })

    return count, max_counts


# Graficar el resultado
def plot_count(count):
    days = range(1, len(count)+1)
    S = [day_data["counts"]["S"] for day_data in count]
    I = [day_data["counts"]["I"] for day_data in count]
    R = [day_data["counts"]["R"] for day_data in count]

    plt.figure(figsize=(10,6))
    plt.plot(days, S, label="Susceptibles")
    plt.plot(days, I, label="Infectados")
    plt.plot(days, R, label="Recuperados")
    plt.xlabel("Día")
    plt.ylabel("Número de personas")
    plt.title("Simulación SIR basada en agentes")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    count, max_info = simulateABM(1000,100,0.7,0.1)
    plot_count(count)
    print("\nMáximos alcanzados:")
    for state in ["S", "I", "R"]:
        print(f"{state}: {max_info[state]['value']} personas el día {max_info[state]['day']}")
