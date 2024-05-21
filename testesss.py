# -*- coding: utf-8 -*-
"""
Created on Tue May 21 10:41:30 2024

@author: ra2257009
"""

import numpy as np
import random

def fitness(x):
    pontuacao = 0
    for i in range(n-1):
        if(x[i] != x[i+1]):
            pontuacao+=1
    return pontuacao
            
n = 25

tamanho_pop = 100 #numero d indivíduos população inicial
populacao = np.random.randint(0,2,size=(tamanho_pop, n)) #número int entre 0 e 1 colocado num vetor de tamanho 25
mutation_prob = 0.05
num_geracoes = 200
for x in range(num_geracoes):
    fitnesses = [fitness(j) for j in populacao]
    maiorFitness = max(fitnesses)
    fitnesses = [f/maiorFitness for f in fitnesses]
    pais = np.array(random.choices(populacao, weights=fitnesses, k=tamanho_pop))
    filhos = []
    for l in range(0, tamanho_pop, 2):
        # Select two parents
        pai, mae = pais[l], pais[l+1]
        # Perform crossover with a probability of 0.5
        if random.random() < 0.5:
            # Create a random crossover point
            crossover_point = random.randint(0, n-1)
            # Create the offspring by combining the genes of the parents
            filhos.append(np.concatenate((pai[:crossover_point], mae[crossover_point:])))
            filhos.append(np.concatenate((mae[:crossover_point], pai[crossover_point:])))
        else:
            # If no crossover occurs, just copy the parents to the offspring
            filhos.append(pai)
            filhos.append(mae)
    for m in range(len(filhos)):
       for o in range(n):
                # Flip the gene with a probability of mutation_prob
                if random.random() < mutation_prob:
                    filhos[m][o] = 1 - filhos[m][o]
    populacao = filhos

best_solution = max(populacao, key=fitness)

# Print the best solution
print(best_solution)