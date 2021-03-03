"""

Kemal Caner Hacıoğlu 

Okul no:2013210217005
    
Konu:Simulating Anneling yöntemi ile özellik seçimi 
    
"""
from __future__ import print_function, division 
import numpy as np
import numpy.random as rn
from sklearn.metrics import accuracy_score   
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings("ignore")
from numpy import mean



#veri setini import etme
#ayrı csv dosyalarını  birleştirme
dfs = []
for label in ['0', '1', '2', '3']:
    dfs.append(pd.read_csv(label + '.csv'))
for df in dfs:
    df.columns = list(range(len(df.columns)))
    data = pd.concat([df for df in dfs], axis=0).reset_index(drop=True)
    liste=[str(x) for x in range(65)]
#bağımlı değişken=y, bağımsız değişkenler=X    
data.columns=liste 
X=data.drop(["64"],axis=1)
y=data["64"].values

#random örneklem oluşturan fonksiyon
def random_start():
    #Random örneklem
    sample_arr = [True, False]
    # Boyutu 64 olan bir random numpy array oluşturur 
    bool_arr = np.random.choice(sample_arr, size=64)
    
    #oluşturulan array'e göre verisetindeki kısımdan bir dataframe oluşturulur
    XStart=X.loc[:,bool_arr] 
    return XStart

#minimize edilecek fonksiyon, 1/accuracy,doğruluğu maksimize etmek
def f(X,y): 
    #minimize edilecek fonksiyon,doğruluk değeri maks yapılmak istendiği için 1/meanScore alınır 
    model= RandomForestClassifier(random_state=1) 
    #model= KNeighborsClassifier() #model
    cv = KFold(n_splits=5, random_state=1, shuffle=True) #Kfold yöntemi ile veri test,train olarak bölütlere ayrılır
     #çapraz doğrulama sonucu elde edilen doğruluk değerleri bulunur. Bu şekilde modelin başarısı ölçülür
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    meanScore=mean(scores) #doğruluk
    return 1/meanScore #doğruluğun artmasını istediğimiz için bu şekilde yazıldı

#oluşturulan numpy tipinde veriyi dataframe'e dönüştürmek için
def newDf(X,new):
    xstryeni = [str(i) for i in new]
    newDF=X.loc[:,xstryeni]
    return newDF
def cost_function(X,y):
    
    return f(X,y)

#kabuledilebilir olasılıksal değeri hesap eden fonksiyon
def acceptance_probability(cost, new_cost, temperature):
    if new_cost < cost:
        return 1
    else:
        p = np.exp(- (new_cost - cost) / temperature)
        return p

#sıcaklık değerini hesap eden fonksiyon
def temperature(fraction):
    return max(0.01, min(1, 1 - fraction))


#verinin tanımlı olacağı aralık belirlenir
interval=(0,63)
a, b = interval

#ilk örneklem random oluşturulur
state = random_start().columns.to_numpy() 
#dataframe çevrilir
stateDF=newDf(X,state)
#maliyeti hesaplanır
cost = cost_function(stateDF,y)

yeni=[]

#adım sayısı
maxsteps=5

#simulating anneling kısmı
for step in range(maxsteps):
    uzunluk=len(state)
    for i in range(0,uzunluk):
                fraction=step / float(maxsteps)
                T = max(0.01, min(1, 1 - fraction))
                fraction=1
                #yeni komşu şeçimi işlemi
                amplitude = (max(interval) - min(interval)) * fraction / 1
                delta = (-amplitude/2.) + amplitude * rn.random_sample() 
                # komşu durumun belirli bir aralığın dışına çıkılmaması için ayrkırı bir kolon numarası sınıra çekildi
                yeni.append(round(max(min(int(state[i])+delta, b), a))) 
                new_state= np.unique(yeni) #birbirini tekrar eden kolonlar olmaması için unique olanlar alındı
                newDF = newDf(X,new_state) # yeni(komşu) durum için dataframe oluşturuldu
                new_cost = cost_function(newDF,y)#yeni durumun maliyeti
    if acceptance_probability(cost, new_cost, T) > rn.random(): #olasılık değerine göre seçim
                state, cost = new_state, new_cost
              
                

print("Feature Sayısı:",len(state) ,"\n\nFeatures:",state,"\n\nDoğruluk değeri  :",1/cost)

    

