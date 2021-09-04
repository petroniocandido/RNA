import numpy as np
from FuncoesAtivacao import sigmoide, identidade
from FuncoesCusto import mse

def treinar(X, Y, **kwargs):
  taxa_aprendizado = kwargs.get("taxa_aprendizado", 0.02)
  numero_epocas = kwargs.get("numero_epocas", 100)
  batch = kwargs.get("batch", 100)
  funcao_custo = kwargs.get("funcao_custo", mse)
  funcao_ativacao = kwargs.get("funcao_ativacao", sigmoide)
  inicializacao_media = kwargs.get("inicializacao_media", 0)
  inicializacao_desvio = kwargs.get("inicializacao_desvio", 0.5)


  #nº de variáveis de entrada
  num_in = X.shape[1]

  #nº de variáveis de saída
  num_out = Y.shape[1]

  #nº de instâncias
  num_inst = X.shape[0] 
  
  #Inicialização dos parâmetros
  W = [np.random.normal(inicializacao_media,inicializacao_desvio,num_in) for k in range(num_out)]
  B = [np.random.normal(inicializacao_media,inicializacao_desvio,1) for k in range(num_out)]

  log_erros = []

  for epocas in np.arange(numero_epocas):

    for z in np.arange(batch):

      i = np.random.randint(0,num_inst)

      erros = []
    
      for saida in range(num_out):

        net = W[saida].dot(X[i,:]) + B[saida]  # WX + B
        
        estimativa = funcao_ativacao(net)  # Ŷ = sigmoide(net)
        
        erros.append(funcao_custo(Y[i,saida],estimativa))  # Armazena o erro
        
        # Regra Delta - O valor de atualização dos pesos
        delta = funcao_custo(Y[i,saida],estimativa, derivada=True) * funcao_ativacao(net, derivada = True)

        #Atualiza os pesos (somo cada peso com seu delta)
        W[saida] += taxa_aprendizado*-delta*-X[i,:]
          
        #Atualização do viés
        B[saida] += taxa_aprendizado*-delta

    log_erros.append(np.mean(erros))   # Retorna o histórico de erros para visualização

  return W, B, log_erros


def treinar_batch(X, Y, **kwargs):
  taxa_aprendizado = kwargs.get("taxa_aprendizado", 0.02)
  numero_epocas = kwargs.get("numero_epocas", 100)
  batch = kwargs.get("batch", 100)
  funcao_custo = kwargs.get("funcao_custo", mse)
  funcao_ativacao = kwargs.get("funcao_ativacao", sigmoide)
  inicializacao_media = kwargs.get("inicializacao_media", 0)
  inicializacao_desvio = kwargs.get("inicializacao_desvio", 0.5)

  num_in = X.shape[1]
  num_out = Y.shape[1]
  num_inst = X.shape[0] 
  W = [np.random.normal(inicializacao_media,inicializacao_desvio,num_in) for k in range(num_out)]
  B = [np.random.normal(inicializacao_media,inicializacao_desvio,1) for k in range(num_out)]
  log_erros = []

  for epocas in np.arange(numero_epocas):
    x_avg = np.zeros(num_in)
    delta_avg = np.zeros(num_out)
    
    erros = []    
    for z in np.arange(batch):
      i = np.random.randint(0,num_inst)
      x_avg += X[i,:]
      for saida in range(num_out):
        net = W[saida].dot(X[i,:]) + B[saida]  # WX + B
        estimativa = sigmoide(net)  # Ŷ = sigmoide(net)
        erros.append(funcao_custo(Y[i,saida],estimativa))  # Armazena o erro
        
        # Regra Delta - O valor de atualização dos pesos
        delta = funcao_custo(Y[i,saida],estimativa, derivada=True) * funcao_ativacao(net, derivada = True)
        delta_avg[saida] += delta
    x_avg /= batch
    delta_avg /= batch
    for saida in range(num_out):
      W[saida] += taxa_aprendizado*-delta_avg[saida]*-x_avg
      B[saida] += taxa_aprendizado*-delta_avg[saida]

    log_erros.append(np.mean(erros))   # Retorna o histórico de erros para visualização

  return W, B, log_erros

def classificacao(x,W,B, **kwargs):
  funcao_ativacao = kwargs.get("funcao_ativacao", sigmoide)
  num_inst = x.shape[0]
  num_in = x.shape[1]
  num_out = len(W)
  ret = []
  for i in range(num_inst):
    tmp = [funcao_ativacao(W[modelo].dot(x[i,:]) + B[modelo])[0] for modelo in np.arange(num_out)]
    tmp = np.where(tmp == max(tmp), 1, 0).argmax()
    ret.append(tmp)
  return np.array(ret)

def regressao(x,W,B,**kwargs):
  funcao_ativacao = kwargs.get("funcao_ativacao", identidade)
  num_inst = x.shape[0]
  num_in = x.shape[1]
  num_out = len(W)
  ret = []
  for i in range(num_inst):
    tmp = [funcao_ativacao(W[modelo].dot(x[i,:]) + B[modelo])[0] for modelo in np.arange(num_out)]
    tmp = np.where(tmp == max(tmp), 1, 0).argmax()
    ret.append(tmp)
  return np.array(ret)
