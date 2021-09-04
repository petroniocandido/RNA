import numpy as np
from RNA.FuncoesAtivacao import sigmoide, identidade
from RNA.FuncoesCusto import mse


def treinar(X, Y, **kwargs):
  taxa_aprendizado = kwargs.get("taxa_aprendizado", 0.02)
  numero_epocas = kwargs.get("numero_epocas", 100)
  batch = kwargs.get("batch", 100)
  funcao_custo = kwargs.get("funcao_custo", mse)
  inicializacao_media = kwargs.get("inicializacao_media", 0)
  inicializacao_desvio = kwargs.get("inicializacao_desvio", 0.5)
  ativacao_camada1 = kwargs.get("ativacao_camada1", sigmoide)
  ativacao_camada2 = kwargs.get("ativacao_camada2", sigmoide)
  num_in = X.shape[1]
  num_neuron = kwargs.get("num_neuron", num_in)
  num_out = Y.shape[1]
  num_inst = X.shape[0] 

  W1 = np.random.normal(inicializacao_media,inicializacao_desvio,(num_neuron, num_in))
  B1 = np.random.normal(inicializacao_media,inicializacao_desvio,(num_neuron,1))

  W2 = np.random.normal(inicializacao_media,inicializacao_desvio,(num_out, num_neuron))
  B2 = np.random.normal(inicializacao_media,inicializacao_desvio,(num_out, 1))

  log_erros = []

  for epocas in np.arange(numero_epocas):

    for z in np.arange(batch):
      i = np.random.randint(0,num_inst)

      #################################
      # Forward
      #################################

      net1 = np.zeros(num_neuron)
      saidas1 = np.zeros(num_neuron)

      for neuronio in range(num_neuron):
        net = W1[neuronio].dot(X[i,:]) + B1[neuronio]
        saidas1[neuronio] = ativacao_camada1(net)
        net1[neuronio] = net

      #saidas1 = np.array(saidas1).flatten()

      net2 = np.zeros(num_out)
      saidas2 = np.zeros(num_out)

      for neuronio in range(num_out):
        net = W2[neuronio].dot(saidas1) + B2[neuronio]
        saidas2[neuronio] = ativacao_camada2(net)
        net2[neuronio] = net

      #saidas2 = np.array(saidas2).flatten()

      #################################
      # Backward
      #################################

      # Camada de Saída

      erros = []
      delta2 = np.zeros(num_out)
      for neuronio in range(num_out):
        erros.append(funcao_custo(Y[i,neuronio], saidas2[neuronio]))
        delta2[neuronio] = funcao_custo(Y[i,neuronio], saidas2[neuronio], derivada=True) * ativacao_camada2(net2[neuronio], derivada=True)
        W2[neuronio] += taxa_aprendizado*-delta2[neuronio]*-saidas1
        B2[neuronio] += taxa_aprendizado*-delta2[neuronio]

      # Camada de Entrada
      
      for neuronio in range(num_neuron):
        delta1 = ativacao_camada1(net1[neuronio], derivada=True) * np.sum([delta2[j] * W2[j][neuronio] for j in range(num_out)])
        W1[neuronio] += taxa_aprendizado*-delta1*-X[i,:]
        B1[neuronio] += taxa_aprendizado*-delta1

      log_erros.append(np.mean(erros))
  return W1, B1, W2, B2, log_erros


def classificacao(x, W1, B1, W2, B2, **kwargs):
  ativacao_camada1 = kwargs.get("ativacao_camada1", sigmoide)
  ativacao_camada2 = kwargs.get("ativacao_camada2", sigmoide)
  num_in = len(W1)
  num_out = len(W2)
  num_inst = x.shape[0]
  num_neuron = kwargs.get("num_neuron", num_in)
  ret = []
  for i in range(num_inst):
    
    saidas1 = np.zeros(num_neuron)
    for neuronio in range(num_in):
      saidas1[neuronio] = ativacao_camada1(W1[neuronio].dot(x[i,:]) + B1[neuronio])

    saidas2 = np.zeros(num_out)
    for neuronio in range(num_out):
      saidas2[neuronio] = ativacao_camada2(W2[neuronio].dot(saidas1) + B2[neuronio])

    ret.append(np.argmax(np.where(saidas2 == max(saidas2), 1, 0)))
  return ret


def regressao(x, W1, B1, W2, B2, **kwargs):
  ativacao_camada1 = kwargs.get("ativacao_camada1", sigmoide)
  ativacao_camada2 = kwargs.get("ativacao_camada2", sigmoide)
  num_in = len(W1)
  num_out = len(W2)
  num_inst = x.shape[0]
  num_neuron = kwargs.get("num_neuron", num_in)
  ret = []
  for i in range(num_inst):
    
    saidas1 = np.zeros(num_neuron)
    for neuronio in range(num_in):
      saidas1[neuronio] = ativacao_camada1(W1[neuronio].dot(x[i,:]) + B1[neuronio])

    saidas2 = np.zeros(num_out)
    for neuronio in range(num_out):
      saidas2[neuronio] = ativacao_camada2(W2[neuronio].dot(saidas1) + B2[neuronio])

    ret.append(saidas2)
  return ret
