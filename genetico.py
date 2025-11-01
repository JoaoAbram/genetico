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


def calcular_distancia_total(caminho, pontos):
    distancia_total = 0
    
    for i in range(len(caminho)):
        ponto_atual = caminho[i]
        
        ponto_proximo = caminho[0] if (i + 1) == len(caminho) else caminho[i+1]
        
        coord_atual = pontos[ponto_atual]
        coord_proxima = pontos[ponto_proximo]
        
        distancia_total += np.linalg.norm(coord_atual - coord_proxima)
        
    return distancia_total

# Agora, a função de aptidão (fitness)
def calcular_aptidao(caminho, pontos):
    distancia = calcular_distancia_total(caminho, pontos)
    
    if distancia == 0:
        return float('inf') 
        
    return 1.0 / distancia

# --- Funções de ajuda para o Indivíduo ---

def criar_individuo_aleatorio(n_pontos):
    individuo = random.sample(range(n_pontos), n_pontos)
    return individuo

# --- Testando o que fizemos ---
print("\n--- Teste das novas funções ---")

# Vamos usar o cenário do círculo (pontos_circulares) que já criamos
print(f"Usando {N_PONTOS} pontos do cenário circular.")

caminho_ordenado = list(range(N_PONTOS)) 
dist_ordenada = calcular_distancia_total(caminho_ordenado, pontos_circulares)
aptidao_ordenada = calcular_aptidao(caminho_ordenado, pontos_circulares)

print(f"Caminho Ordenado: {caminho_ordenado}")
print(f"Distância (Ordenado): {dist_ordenada:.2f}")
print(f"Aptidão (Ordenado): {aptidao_ordenada:.4f}")
print("-" * 20)

caminho_aleatorio = criar_individuo_aleatorio(N_PONTOS)
dist_aleatoria = calcular_distancia_total(caminho_aleatorio, pontos_circulares)
aptidao_aleatoria = calcular_aptidao(caminho_aleatorio, pontos_circulares)

print(f"Caminho Aleatório: {caminho_aleatorio}")
print(f"Distância (Aleatório): {dist_aleatoria:.2f}")
print(f"Aptidão (Aleatório): {aptidao_aleatoria:.4f}")

# --- PASSO 3: Operadores Genéticos (População, Seleção, Cruzamento, Mutação) ---

# --- 3.1: Parâmetros do Algoritmo ---
# Vamos definir nossas "regras do jogo" aqui
TAMANHO_POPULACAO = 50     # Quantos indivíduos (caminhos) em cada geração
TAXA_ELITISMO = 0.1        # 10% dos melhores vão direto para a próxima geração
TAXA_MUTACAO = 0.01        # 1% de chance de um indivíduo sofrer mutação

# --- 3.2: Criar a População Inicial ---
def criar_populacao_inicial(n_individuos, n_pontos):
    """Cria uma lista de indivíduos aleatórios."""
    populacao = []
    for _ in range(n_individuos):
        populacao.append(criar_individuo_aleatorio(n_pontos))
    return populacao

# --- 3.3: Seleção (Explicação vale 1 ponto) ---
def selecao_por_torneio(populacao, aptidoes, k=3):
    """
    Seleciona um 'pai' usando o método do Torneio.
    k = quantos indivíduos vão 'batalhar' no torneio.
    """
    # Sorteia k índices aleatórios da população
    indices_torneio = random.sample(range(len(populacao)), k)
    
    # Encontra o índice do vencedor (maior aptidão)
    indice_vencedor = -1
    melhor_aptidao = -1.0 # -1 (qualquer aptidão real será maior)
    
    for i in indices_torneio:
        if aptidoes[i] > melhor_aptidao:
            melhor_aptidao = aptidoes[i]
            indice_vencedor = i
            
    # Retorna o indivíduo vencedor (o 'pai' ou 'mãe')
    return populacao[indice_vencedor]

# --- 3.4: Cruzamento (Explicação vale 2 pontos) ---
def cruzamento_ordenado(pai1, pai2):
    """
    Gera um filho usando Crossover Ordenado (OX).
    Garante que o filho é uma permutação válida.
    """
    tamanho = len(pai1)
    
    # 1. Sorteia dois pontos de corte (início e fim do 'miolo')
    inicio, fim = sorted(random.sample(range(tamanho), 2))
    
    # 2. Cria o filho com 'None' (vazio)
    filho = [None] * tamanho
    
    # 3. Copia o 'miolo' do pai1 para o filho
    miolo_pai1 = pai1[inicio : fim + 1]
    filho[inicio : fim + 1] = miolo_pai1
    
    # 4. Pega os genes do pai2 que *não* estão no miolo
    ponteiro_pai2 = 0
    ponteiro_filho = 0
    
    while None in filho: # Enquanto o filho não estiver completo
        # Acha a próxima posição vazia no filho
        if ponteiro_filho == inicio:
            ponteiro_filho = fim + 1 # Pula o miolo que já copiamos
        
        # Pega o gene do pai2
        gene_pai2 = pai2[ponteiro_pai2]
        ponteiro_pai2 += 1 # Move para o próximo gene do pai2
        
        # Se esse gene do pai2 *NÃO* estiver no miolo, adiciona no filho
        if gene_pai2 not in miolo_pai1:
            filho[ponteiro_filho] = gene_pai2
            ponteiro_filho += 1 # Move para o próximo espaço vazio do filho
            
    return filho

# --- 3.5: Mutação (Explicação vale 1 ponto) ---
def mutacao_por_troca(individuo, taxa_mutacao):
    """
    Aplica a mutação de troca (Swap Mutation).
    Com uma chance (taxa_mutacao), troca dois genes de lugar.
    """
    # Verifica se a mutação "acontece"
    if random.random() < taxa_mutacao:
        # Sorteia duas posições (índices) aleatórias para trocar
        pos1, pos2 = random.sample(range(len(individuo)), 2)
        
        # Faz a troca (swap)
        individuo[pos1], individuo[pos2] = individuo[pos2], individuo[pos1]
        
    return individuo

# --- PASSO 4: O Motor da Evolução e Análise ---

import time # Vamos precisar para o bônus (medir o tempo)

def rodar_ga(pontos, n_epocas):
    """
    Função principal que executa o Algoritmo Genético.
    """
    print(f"\nIniciando GA para {len(pontos)} pontos por {n_epocas} épocas...")
    
    # Pega o número de pontos do array de coordenadas
    n_pontos = len(pontos)
    
    # 1. CRIA A POPULAÇÃO INICIAL
    populacao = criar_populacao_inicial(TAMANHO_POPULACAO, n_pontos)
    
    # Guarda o melhor caminho e distância já encontrados (começa com infinito)
    melhor_distancia_global = float('inf')
    melhor_caminho_global = None
    
    # Lista para guardar a melhor distância de CADA época (para o gráfico)
    historico_distancias = []

    # --- INÍCIO DO LOOP DE EVOLUÇÃO ---
    for epoca in range(n_epocas):
        
        # 2. AVALIAÇÃO (Calcula aptidão de todos)
        aptidoes = []
        for individuo in populacao:
            aptidoes.append(calcular_aptidao(individuo, pontos))
            
        # Encontra a melhor distância *desta* época
        melhor_distancia_epoca = 1.0 / max(aptidoes) # max(aptidões) = 1 / min(dist)
        historico_distancias.append(melhor_distancia_epoca)
        
        # Atualiza o melhor global (se o desta época for melhor)
        if melhor_distancia_epoca < melhor_distancia_global:
            melhor_distancia_global = melhor_distancia_epoca
            # Encontra o índice do melhor e guarda o caminho
            indice_melhor = np.argmax(aptidoes) 
            melhor_caminho_global = populacao[indice_melhor]
            
            # Print para acompanhar o progresso
            print(f"Época {epoca}: Nova melhor distância = {melhor_distancia_global:.2f}")

        # 3. CRIA A NOVA GERAÇÃO
        nova_populacao = []
        
        # 3.1 Elitismo: Passa os melhores direto
        n_elite = int(TAMANHO_POPULACAO * TAXA_ELITISMO)
        
        # Pega os índices dos 'n_elite' melhores (os que têm maior aptidão)
        indices_ordenados = np.argsort(aptidoes)[::-1] # [::-1] inverte (pega os maiores)
        
        for i in range(n_elite):
            indice_elite = indices_ordenados[i]
            nova_populacao.append(populacao[indice_elite])
            
        # 3.2 Preenchimento: Cria o resto com cruzamento e mutação
        n_filhos = TAMANHO_POPULACAO - n_elite
        
        for _ in range(n_filhos):
            # Seleciona os pais
            pai1 = selecao_por_torneio(populacao, aptidoes)
            pai2 = selecao_por_torneio(populacao, aptidoes)
            
            # Cria o filho
            filho = cruzamento_ordenado(pai1, pai2)
            
            # Aplica (ou não) a mutação
            filho_mutado = mutacao_por_troca(filho, TAXA_MUTACAO)
            
            # Adiciona o filho na nova geração
            nova_populacao.append(filho_mutado)
            
        # 4. ATUALIZA A POPULAÇÃO
        # A nova geração vira a população atual para a próxima época
        populacao = nova_populacao
        
    # --- FIM DO LOOP DE EVOLUÇÃO ---
    
    print(f"GA finalizado! Melhor distância encontrada: {melhor_distancia_global:.2f}")
    
    # Retorna o melhor caminho e o histórico para o gráfico
    return melhor_caminho_global, historico_distancias

def plotar_solucao(pontos, caminho, historico_dist, titulo):
    """
    Função para plotar os resultados:
    1. O gráfico de desempenho (distância vs. época)
    2. O caminho final encontrado
    """
    
    # --- Gráfico 1: Desempenho (Evolução) ---
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(historico_dist)
    plt.title(f"Desempenho (Época vs. Distância)\n{titulo}")
    plt.xlabel("Época")
    plt.ylabel("Melhor Distância")
    
    # --- Gráfico 2: Solução (Caminho Final) ---
    plt.subplot(1, 2, 2)
    
    # Coloca os pontos do caminho na ordem correta
    pontos_ordenados = pontos[caminho]
    
    # Adiciona o primeiro ponto no final para fechar o ciclo
    pontos_ordenados = np.vstack([pontos_ordenados, pontos_ordenados[0]])
    
    # Desenha as linhas (o caminho)
    plt.plot(pontos_ordenados[:, 0], pontos_ordenados[:, 1], 'o-') # 'o-' = bolinha com linha
    
    # Coloca os números das cidades
    for i, (x, y) in enumerate(pontos):
        plt.text(x, y, str(i), color="red", fontsize=10)
        
    plt.title(f"Solução Final Encontrada\n{titulo}")
    plt.xlabel("X")
    plt.ylabel("Y")
    
    plt.tight_layout() # Ajusta o layout para não sobrepor
    plt.show() # MOSTRA A JANELA COM OS DOIS GRÁFICOS


# --- BLOCO PRINCIPAL DE EXECUÇÃO ---
# É aqui que a gente "aperta o play"
if __name__ == "__main__":
    
    # --- Parâmetros Gerais ---
    # Vamos aumentar o número de pontos para o teste ser mais legal
    N_PONTOS_TESTE = 20
    N_EPOCAS = 100 # Quantas gerações vamos rodar
    
    # Sobrescreve as variáveis do Passo 1 (opcional, mas bom para testar)
    pontos_aleatorios = gerar_pontos_aleatorios(N_PONTOS_TESTE)
    pontos_circulares = gerar_pontos_circulares(N_PONTOS_TESTE)

    # --- CENÁRIO 1: ALEATÓRIO (Vale 2,5 pontos) ---
    print("--- INICIANDO CENÁRIO ALEATÓRIO ---")
    
    # Medindo o tempo (BÔNUS)
    tempo_inicio_aleatorio = time.time()
    
    melhor_caminho_aleatorio, historico_aleatorio = rodar_ga(pontos_aleatorios, N_EPOCAS)
    
    tempo_fim_aleatorio = time.time()
    print(f"Tempo de execução (Aleatório): {tempo_fim_aleatorio - tempo_inicio_aleatorio:.2f} segundos")
    
    # Plotar os resultados do cenário aleatório
    plotar_solucao(pontos_aleatorios, melhor_caminho_aleatorio, historico_aleatorio, "Cenário Aleatório")

    # --- CENÁRIO 2: CIRCULAR (Vale 2,5 pontos) ---
    print("\n--- INICIANDO CENÁRIO CIRCULAR ---")
    
    # Medindo o tempo (BÔNUS)
    tempo_inicio_circular = time.time()
    
    melhor_caminho_circular, historico_circular = rodar_ga(pontos_circulares, N_EPOCAS)
    
    tempo_fim_circular = time.time()
    print(f"Tempo de execução (Circular): {tempo_fim_circular - tempo_inicio_circular:.2f} segundos")

    # Plotar os resultados do cenário circular
    plotar_solucao(pontos_circulares, melhor_caminho_circular, historico_circular, "Cenário Circular")
    
    print("\n--- FIM DO TRABALHO ---")