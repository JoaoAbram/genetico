import numpy as np
import random
import matplotlib.pyplot as plt

def gerar_pontos_aleatorios (n_pontos, largura_max=100, altura_max=100):
    pontos = np.random.rand(n_pontos, 2)
    pontos [:, 0] *= largura_max
    pontos [:, 1] *= altura_max
    return pontos

def gerar_pontos_circulares (n_pontos, raio=50, centro_x=50, centro_y=50):
    pontos = []
    for i in range(n_pontos):
        angulo = (2 * np.pi * i) / n_pontos
        x = centro_x + raio * np.cos(angulo)
        y = centro_y + raio * np.sin(angulo)
        pontos.append((x,y))
    return np.array (pontos)
    
N_PONTOS = 10

pontos_aleatorios = gerar_pontos_aleatorios(N_PONTOS)

pontos_circulares = gerar_pontos_circulares(N_PONTOS)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title ("Cenario 1: Aleatorio")
plt.scatter(pontos_aleatorios[:, 0], pontos_aleatorios[:, 1])
for i, (x,y) in enumerate(pontos_aleatorios):
    plt.text(x, y, str(i))

plt.subplot(1, 2, 2)
plt.title("Cenário 2: Circulo")
plt.scatter(pontos_circulares[:, 0],pontos_circulares[:, 1])
for i, (x,y) in enumerate(pontos_circulares):
    plt.text(x, y, str(i))

plt.show

