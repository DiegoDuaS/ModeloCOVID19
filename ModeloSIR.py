from matplotlib import pyplot as plt
import numpy as np
from scipy.integrate import odeint


def modelo_SRI(y, t, beta, gamma):
  """
  Ecuaciones diferenciales del modelo SIR
  """
  S, I, R = y
  N = S + I + R  # Población total
  #2.Flujos
  # Las ecuaciones diferenciales se calculan en la función modelo_SRI
  #Tasa de infección:
  dSdt = -beta * S * I / N           # Susceptibles que se infectan
  dIdt = beta * S * I / N - gamma * I # Nuevos infectados - recuperados
  #Tasa de recuperación:
  dRdt = gamma * I                   # Infectados que se recuperan
  return dSdt, dIdt, dRdt

def run_simulation(num_pers_suseptibles, num_pers_infectadas, num_pers_recuperadas, parametros_de_infeccion, parametros_de_recuperacion, dias): 
  #1.recursos
  poblacion_total = num_pers_suseptibles + num_pers_infectadas + num_pers_recuperadas 
  susceptibles = num_pers_suseptibles
  infectadas = num_pers_infectadas
  recuperadas = num_pers_recuperadas
  print("La epidemia tiene las siguientes caracteríticas inciales")
  print(f"\tpoblacion_total: {poblacion_total}\n\tsusceptibles: {susceptibles}\n\tinfectadas: {infectadas}\n\trecuperadas :{recuperadas}\n\tdías:{dias}\n\tparametro de infección: {parametros_de_infeccion}\n\tparametro de recuperación: {parametros_de_recuperacion}\n")


  # Las condiciones iniciales
  y0 = [susceptibles, infectadas, recuperadas]
  
  # Tiempo a evluar
  dias_por_evaluar = np.linspace(0, dias, dias + 1)  # +1 para incluir el día final
  
  # nos ayudamos de una libería para evaluar la ecuación diferencias
  sol = odeint(modelo_SRI, y0, dias_por_evaluar, args=(parametros_de_infeccion, parametros_de_recuperacion))
  
  # Calcular R0
  r0 = parametros_de_infeccion / parametros_de_recuperacion
  
  # Calcular tasa de cambio inicial
  tasa_de_cambio = parametros_de_infeccion * (susceptibles/poblacion_total) * infectadas - parametros_de_recuperacion * infectadas
      
  print("R0:", r0)
  print("Tasa de cambio inicial:", tasa_de_cambio)
  calcular_metricas_adicionales(sol,parametros_de_infeccion,parametros_de_recuperacion,poblacion_total)
  graficacion(sol,dias_por_evaluar)
  return sol, r0, tasa_de_cambio

def graficacion(sol, dias_por_evaluar): 
  plt.figure(figsize=(10,5))
  
  #varaibles de la ecuacion diferencial
  S = sol[:, 0]  # Susceptibles
  I = sol[:, 1]  # Infectados
  R = sol[:, 2]  # Recuperados

  print("Datos finales despues de " +str(round(dias_por_evaluar[-1])) + " días")
  print(f"\n\tsusceptibles: {round(S[-1])}\n\tinfectadas: {round(I[-1])}\n\trecuperadas :{round(R[-1])}")

  #integrarlas a la grafica
  plt.plot(dias_por_evaluar,S, color='green', label='Susceptibles')
  plt.plot(dias_por_evaluar,I, color='red', label='Infectados')
  plt.plot(dias_por_evaluar,R, color='blue', label='Recuperados')
  plt.xlabel('Días')
  plt.ylabel('Número de Personas')
  plt.title('Modelo SIR - Evolución de la Epidemia')
  plt.legend()
  plt.grid(True)
  plt.show()
  

def calcular_metricas_adicionales(sol, beta, gamma, poblacion_total):
  """Calcula métricas epidemiológicas adicionales"""
  # Pico de la epidemia
  pico_infectados = np.max(sol[:, 1])
  dia_pico = np.argmax(sol[:, 1])
  # Tamaño final de la epidemia  
  infectados_totales = poblacion_total - sol[-1, 0]
  print(f"pico_infectados: {pico_infectados}\ndia_pico: {dia_pico}\ninfectados_totales: {infectados_totales}")

if __name__ == "__main__":
  run_simulation(99900, 100, 0, 0.3, 0.1, 365)