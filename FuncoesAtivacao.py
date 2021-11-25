import numpy as np

def identidade(x, derivada = False):
  if not derivada:
    return x
  else:
    return 1

def degrau(x, derivada = False):
  if not derivada:
    return np.where(x > 0, 1, 0)
  else:
    return np.where(x > 0, 1, 0)

def sigmoide(x, derivada = False):
  if not derivada:
    return 1./(1. + np.exp(-x))
  else:
    y = sigmoide(x)
    return y*(1-y)

def relu(x, derivada = False):
  if not derivada:
    return np.maximum(x, 0)
  else:
    return np.where(np.maximum(x, 0) > 0, 1, 0) 
