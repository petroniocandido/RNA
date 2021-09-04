

def mse(y, yl, derivada = False):
  if not derivada:
    return ((y-yl)**2)/2
  else:
    return y-yl
