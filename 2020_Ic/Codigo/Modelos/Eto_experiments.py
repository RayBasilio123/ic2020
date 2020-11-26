import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib as plt
import matplotlib.pyplot as plt
from math import nan
from Tratamento.variaveis import latitude_2,  altitude_2, sigma, G, Gsc
from Tratamento.Eto_generator import gera_serie
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf



def get_x2(df,lista,lags,Eto,lags_eto):
  lags=lags
  lista_aux=[]
  lista_aux2=[]
  max_lag=0;
  data = pd.DataFrame()
  eta_nois = pd.DataFrame()
  eta_nois['Data']=df['Data']
  for coluna in lista:
    data[coluna] = df[coluna]       

  for i in range(len(lags)):
    for j in range(len(lags[i])):
      lista_aux = data.iloc[:,i].tolist()
      for displacement in range((lags[i][j])):
        if max_lag<(lags[i][j]):
          max_lag=(lags[i][j])
        del lista_aux[len(lista_aux)-1]
        lista_aux.insert(0,nan)
      eta_nois[((data.iloc[:,i]).name)+("_t-")+str(lags[i][j])]=(lista_aux)  
  
  for i in range(len(lags_eto)):
    lista_aux2=df[Eto].iloc[:,0].tolist()
    for displacement in range((lags_eto[i])):
      if max_lag<(lags_eto[i]):
        max_lag=(lags_eto[i])
      del lista_aux2[len(lista_aux2)-1]
      lista_aux2.insert(0,nan)
    eta_nois[((df[Eto].iloc[:,0]).name)+("_t-")+str(lags_eto[i])]=lista_aux2        
  return eta_nois,max_lag,lista,lags,Eto,lags_eto;



def get_x30(df,lista, Eto):
  ix = []
  idx =  [i for i in np.arange(1, 31)]
  for i in range(len(lista)):
    ix.append(idx)

  resultado = get_x2(df, lista, ix, Eto, idx)
  return resultado

  


def pacf_acf(df_patricia,atributeP):
  for i in (atributeP):
    plot_acf(df_patricia[i],lags=60,title="Autocorrelation "+i)
    plot_pacf(df_patricia[i],lags=60,title= "Partial Autocorrelation "+i)
  plt.show();


  

from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.ensemble import ExtraTreesRegressor

def selecao_recursos(df) :

  # from scipy import stats
  # dfp=df_patricia.drop("Data", axis=1)
  # Tmax	Tmin	I	Tmean	UR	V	J	Eto
  #  0      1   2   3    4  5 6   7
  df30=df30.drop("Data",axis=1)
  df30=df30.dropna()
  array1 = df30.values
  df= df.drop("Data",axis=1)

  array2 = df["Eto"]
  X =array1[:,0:31]
  Y = array2
  # feature extraction
  test = SelectKBest(score_func=f_regression, k=4)
  fit = test.fit(X, Y)
  # # summarize scores
  set_printoptions(precision=3)
  print(dfp.columns[:7])
  print("Selecao_univariada",fit.scores_)
  f=fit.scores_
  features = fit.transform(X)
  # summarize selected features
  # print(features[0:5,:])
  model = ExtraTreesRegressor(n_estimators=10)
  model.fit(X, Y)
  print("Importancia",model.feature_importances_)
  
  return f



# # load data

# array = dataframe.values
# X = array[:,0:8]
# Y = array[:,8]
# # feature extraction

# print(model.feature_importances_)