# -*- coding: utf-8 -*-
"""
Created on Tue May 21 10:41:30 2024

@author: ra2257009
"""

import numpy as np
import random

def fitness(x):
    

n = 25

tamanho_pop = 100 #numero d indivíduos população inicial
populacao = np.random.randint(0,2,size=(tamanho_pop, n)) #número int entre 0 e 1 colocado num vetor de tamanho 25
print(populacao)
mutation_prob = 0.05
