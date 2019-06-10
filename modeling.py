#%%

# in this file we continue some features engineering and we start after that the modeling 


import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np 
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

#%%
df_jan_2015_demand=pd.read_pickle("pickle/df_jan_2015_demand.pkl")
df_feb_2015_demand=pd.read_pickle("pickle/df_feb_2015_demand.pkl")
df_mar_2015_demand=pd.read_pickle("pickle/df_mar_2015_demand.pkl")
#%%

df_jan_2016_demand=pd.read_pickle("pickle/df_jan_2016_demand.pkl")
df_feb_2016_demand=pd.read_pickle("pickle/df_feb_2016_demand.pkl")
df_mar_2016_demand=pd.read_pickle("pickle/df_mar_2016_demand.pkl")

#%%
regions=list(df_jan_2015_demand["regions"].unique())
#%%
print(regions)
#%%
df_jan_2016_demand.head()
#%%
windows_size=5
time_serie=pd.DataFrame()
time_serie["taxi_demand"]=df_feb_2016_demand[df_jan_2016_demand["regions"]==2]["taxi_demand"]

error=[0]
true_values=list(time_serie["taxi_demand"])
prediction=[true_values[0]]
for j in range(1,len(true_values)):
    if j < windows_size:
        l=true_values[0:j]
    else:
        l=true_values[j-windows_size:j]
    pred=np.mean(l)
    er=abs(pred-true_values[j])
    prediction.append(pred)
    error.append(er)
time_serie['prediction']=prediction
time_serie['error']=error



#%%
def simple_moving_avg(windows_size,region,df):
    time_serie=pd.DataFrame()
    time_serie["taxi_demand"]=df[df["regions"]==region]["taxi_demand"]
    #time_serie["time_bin"]=df[df["regions"]==region]["time_bin"]
    #time_serie.set_index("time_bin")
    error=[0]
    true_values=list(time_serie["taxi_demand"])
    prediction=[true_values[0]]
    for j in range(1,len(true_values)):
        if j < windows_size:
            l=true_values[0:j]
        else:
            l=true_values[j-windows_size:j]
    pred=np.mean(l)
    er=abs(pred-true_values[j])
    prediction.append(pred)
    error.append(er)
    time_serie['prediction']=prediction
    time_serie['error']=error
    mse=np.mean(error)
    return time_serie, mse

#%%
time_serie.head(10)

#%%
windows_size=2
time_serie=pd.DataFrame()
time_serie["taxi_demand"]=df_feb_2016_demand[df_jan_2016_demand["regions"]==2]["taxi_demand"]

error=[0]
true_values=list(time_serie["taxi_demand"])
prediction=[true_values[0]]
for j in range(1,len(true_values)):
    if j < windows_size:
        n=j
        l=true_values[0:j]
    else:
        n=windows_size
        l=true_values[j-windows_size:j]
    p=2*(np.array(l).dot(np.array(range(1,n+1))))/(n*(n+1))
    er=abs(p-true_values[j])
    prediction.append(p)
    error.append(er)
time_serie['prediction']=prediction
time_serie['error']=error
#%%
time_serie=pd.DataFrame()
time_serie["taxi_demand"]=df_feb_2016_demand[df_jan_2016_demand["regions"]==2]["taxi_demand"]
alpha=0.6
error=[0]
true_values=list(time_serie["taxi_demand"])
prediction=[true_values[0]]
for j in range(1,len(true_values)):
   
    p=alpha*true_values[j-1]+(1-alpha)*prediction[j-1]
    #er=abs(p-true_values[j])
    prediction.append(p)
    #error.append(er)
time_serie['prediction']=prediction
#time_serie['error']=error

#%%
def weighted_moving_avg(windows_size,df,region):
    time_serie=pd.DataFrame()
    time_serie["taxi_demand"]=df[df["regions"]==region]["taxi_demand"]

    error=[0]
    true_values=list(time_serie["taxi_demand"])
    prediction=[true_values[0]]
    for j in range(1,len(true_values)):
        if j < windows_size:
            n=j
            l=true_values[0:j]
        else:
            n=windows_size
            l=true_values[j-windows_size:j]
        p=2*(np.array(l).dot(np.array(range(1,n+1))))/(n*(n+1))
        er=abs(p-true_values[j])
        prediction.append(p)
        error.append(er)
    time_serie['prediction']=prediction
    time_serie['error']=error
    return time_serie
#%%
t=weighted_moving_avg(windows_size=5,df=df_feb_2016_demand,region=3)

#%%
time_serie.tail(n=10)

#%%
print(t["error"].mean())

#%%
def expo(df,region,alpha):
    time_serie=pd.DataFrame()
    time_serie["taxi_demand"]=df_feb_2016_demand[df_jan_2016_demand["regions"]==region]["taxi_demand"]
    true_values=list(time_serie["taxi_demand"])
    prediction=[true_values[0]]
    for j in range(1,len(true_values)):
   
        p=alpha*true_values[j-1]+(1-alpha)*prediction[j-1]
        prediction.append(p)
    time_serie['prediction']=prediction
    time_serie['error']=abs(time_serie['taxi_demand']-time_serie['prediction'])
    return time_serie

#%%
t=expo(df=df_mar_2016_demand,region=30,alpha=0.65)

#%%
t.tail(n=5)
#%%
Y=np.abs(np.fft.fft(df_feb_2016_demand[df_jan_2016_demand["regions"]==2]["taxi_demand"].values))

#%%
a=Y.shape[0]
freq = np.abs(np.fft.fftfreq(a, 1))
#%%
print(df_feb_2016_demand.shape)
#%%
print(a)

#%%
plt.figure(figsize = (8, 6))
plt.plot(freq,Y)
#%%
def frequency(df,reg,n):
    Y=np.abs(np.fft.fft(df_feb_2016_demand[df_jan_2016_demand["regions"]==reg]["taxi_demand"].values))
    a=Y.shape[0]
    freq = np.abs(np.fft.fftfreq(a, 1))
    b=Y.argpartition(2)[:n]
    return np.take(Y, b) , np.take(freq, b)

#%%
a=np.array([1, 8, 6, 9, 4])
b=a.argsort()
print(b)

#%%
frequency(df_feb_2016_demand,2,3)

#%%
# construct features for regression models 
df=pd.DataFrame()
df['regions']=df_jan_2016_demand["regions"]
df['target']=df_jan_2016_demand["taxi_demand"]
df['old_demand']=df_jan_2015_demand["taxi_demand"]

#%%
def difference_old(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return diff



#%%
gr=df_jan_2015_demand.groupby("regions")
l=[]
for j in regions: 
    gr_j=gr.get_group(j)
    diff=[0]*1
    for i in range(1, len(gr_j)):
		    value = list(gr_j["taxi_demand"])[i] - list(gr_j["taxi_demand"])[i-1]
		    diff.append(value)
    l=l+diff
df['diff']=l

#%%
df['regions'].equals(df_jan_2015_demand["regions"])

#%%
gr=df_feb_2015_demand.groupby("regions")
#%%
list(gr.get_group(0)["taxi_demand"])[2]

#%%
gr=df_feb_2015_demand.groupby("regions")
l=0
for j in regions:
    l=l+len(gr.get_group(j))

#%%
print(l)

#%%
gr_1=df_feb_2015_demand.groupby("regions")
gr_2=df_jan_2016_demand.groupby("regions")
y=pd.DataFrame()
w=pd.DataFrame()
for u,v in gr_1 :
    y[str(u)] = v['taxi_demand'].values
for k , j in gr_2 :
    w[str(k)] = j['taxi_demand'].values
#%%
plt.figure(figsize = (20, 6))
y.boxplot()
#%%

plt.figure(figsize = (20, 6))
w.boxplot()

#%%
df_feb_2015_demand[df_feb_2015_demand['regions']==2]['taxi_demand'].plot(kind='box')

#%%
df_feb_2016_demand[df_feb_2016_demand['regions']==2]['taxi_demand'].plot(kind='box')

#%%
df_jan_2015_demand['taxi_demand'].equals(df_jan_2016_demand['taxi_demand'])
#%%
def diffrenciation(df_new,df_old,degree):
    gr=df_old.groupby("regions")

    l=[]
    for j,K in gr : 
        diff=[0]*1
        gr_j=gr.get_group(j)
        for i in range(1, len(gr_j)):
		        value = list(k["taxi_demand"])[i] - list(k["taxi_demand"])[i-1]
		        diff.append(value)
        l=l+diff
    df_new['diff']=l
    return df_new
#%%

df_feb_2016_demand['taxi_demand'].equals(df_feb_2015_demand['taxi_demand'])


#%%
print(df_feb_2015_demand.equals(df_feb_2016_demand))
print(df_jan_2015_demand.equals(df_jan_2016_demand))
print(df_mar_2015_demand.equals(df_mar_2016_demand))

#%%
w.equals(y)

#%%
gr=df.groupby("regions")
y=pd.DataFrame()
w=pd.DataFrame()
for u,v in gr :
    y[str(u)] = v["target"].values
    w[str(u)] = v["old_demand"].values

#%%
plt.figure(figsize = (20, 6))
y[y["regions"]==0].plot()
#%%

plt.figure(figsize = (20, 6))
w.boxplot()

#%%

def diffrenciation_old(df_new,df_old,degree): 
    regions=list(df_old["regions"].unique())
    gr=df_old.groupby("regions")
    
    l=[]
    for j in  regions : 
        gr_j=gr.get_group(j)
        diff=[np.mean(list(gr_j["taxi_demand"]))]*degree
        

        for i in range(degree, len(gr_j)):
		        value = list(gr_j["taxi_demand"])[i] - list(gr_j["taxi_demand"])[i-degree]
		        diff.append(value)
        l=l+diff
    
    return l

#%%
df['diff']=diffrenciation_old(df_new=df_jan_2016_demand,df_old=df_jan_2015_demand,degree=1)

#%%
def diffrenciation_new(df_new,df_old,degree): 
    regions=list(df_old["regions"].unique())
    gr=df_old.groupby("regions")
    
    l=[]
    for j in  regions : 
        diff=[]
        gr_j=gr.get_group(j)
        for i in range(0, len(gr_j)-degree):
		        value = list(gr_j["taxi_demand"])[i+degree] - list(gr_j["taxi_demand"])[i]
		        diff.append(value)
        l=l+diff+[np.mean(list(gr_j["taxi_demand"]))]*degree
    
    return l
#%%
df=pd.DataFrame()
df['diff_new']=diffrenciation_new(df_new=df_jan_2016_demand,df_old=df_jan_2015_demand,degree=1)

#%%
def diffrenciation_same(df_new,degree): 
    regions=list(df_new["regions"].unique())
    gr=df_new.groupby("regions")
    l=[]
    for j in  regions : 
        gr_j=gr.get_group(j)
        diff=[np.mean(list(gr_j["taxi_demand"]))]*degree
        for i in range(degree, len(gr_j)):
		        value = list(gr_j["taxi_demand"])[i-degree]
		        diff.append(value)
        l=l+diff
    
    return l

#%%
df["just_before"]=diffrenciation_same(df_new=df_jan_2016_demand,degree=1)

#%%
df.head(n=10)

#%%
X=df.drop(columns=['target'])
y=df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123, shuffle=True)

#%%
xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.05,
                max_depth = 4, alpha = 10, n_estimators = 1000 , eval= 'mae')

#%%
xg_reg.fit(X_train,y_train)

#%%
pickle.dump( df, open( "df", "wb" ) )
#%%
preds = xg_reg.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))


#%%
def expo_frame(df_new,alpha):
    regions=list(df_new["regions"].unique())
    gr=df_new.groupby("regions")
    l=[]
    for j in  regions : 
        gr_j=gr.get_group(j)
        true_values=list(gr_j["taxi_demand"])
        prediction=[true_values[0]]
        for k in range(1,len(true_values)):
        
            p=alpha*true_values[k-1]+(1-alpha)*prediction[k-1]
            prediction.append(p)
        l=l+prediction
    return l 

#%%
def features_engineering_for_regression(df_old, df_new):
    df=pd.DataFrame()
    df['regions']=df_new["regions"]
    df['target']=df_new["taxi_demand"]
    df['old_demand']=df_old["taxi_demand"]
    df['diff_before']=diffrenciation_old(df_new=df_new,df_old=df_old,degree=1)
    df['diff_after']=diffrenciation_new(df_new=df_new,df_old=df_old,degree=1)
    df["just_before"]=diffrenciation_same(df_new=df_new,degree=1)
    df['expo_smothing']=expo_frame(df_new=df_new,alpha=0.6)
    return df 


    



#%%
df=features_engineering_for_regression(df_old=df_jan_2015_demand,df_new=df_jan_2016_demand)


#%%
df.head(20)


#%%
def smothing_zeros(df_new):
    regions=list(df_new["regions"].unique())
    gr=df_new.groupby("regions")
    l=[]
    for j in  regions : 
        gr_j=gr.get_group(j)
        true_values=list(gr_j["taxi_demand"])
        diff=[]
        for k in range(len(true_values)):
            if true_values[k]==0: 
                if  k==0 :
                    for j in range(k,len(true_values)):
                        if true_values[j] != 0 :
                            diff.append(true_values[j]/2)
                            break
                if k!=0 & k != len(true_values)-1:
                    a=0
                    b=0
                    for j in range(k,len(true_values)):
                        if true_values[j] != 0 :
                            a=true_values[j]
                            break
                    for j in range(k,0,-1):
                        if true_values[j] != 0 :
                            b=true_values[j]
                            break
                    diff.append((a+b)/2)

                if k == len(true_values)-1:
                    for j in range(k,0,-1):
                        if true_values[j] != 0 :
                            diff.append(true_values[j]/2)
                            break
            else:
                diff.append(true_values[k])
        l=l+diff

    return l 


                



#%%
X=df.drop(columns=['target']).values
y=df['target'].values


#%%
pred=np.zeros(len(y),dtype=float)
test=np.zeros(len(y))
cv = KFold(n_splits=10, random_state=42, shuffle=True)
for train_index, test_index in cv.split(X):
     X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
     xg_reg.fit(X_train,y_train)
     p=xg_reg.predict(X_test)
     pred[test_index] = p 
     test [test_index] = y_test 








#%%
pred

#%%
test

#%%
mean_squared_error(test,pred)

#%%
xg_reg.fit(X,y)
predi=xg_reg.predict(X)
mean_squared_error(y,predi)


#%%
plt.matshow(df.corr())


#%%
a=expo_frame(df_new=df_jan_2016_demand,alpha=0.6)

#%%
a[:5]

#%%
mean_squared_error(y,a)

#%%

l=smothing_zeros(df_jan_2016_demand)

#%%
df['smoothed_target']=smothing_zeros(df_jan_2016_demand)

#%%
print(len(l))

#%%
df.shape

#%%



#%%
plt.figure(figsize = (8, 6))
plt.plot(df['smoothed_target'])

#%%
plt.figure(figsize = (8, 6))
plt.plot(df['target'])


#%%
df.head(n=20)

#%%
df[df['target']==0][['target','smoothed_target']].head(n=50)

#%%
def MAPE(real,prediction):
    o= abs(np.subtract(real,prediction))/real
    return (100*sum(o))/len(o)

#%%
MAPE(list(df['smoothed_target']),a)


#%%
[1,23]-[1,2]

#%%
list(set([1,23]) - set([1,2]))

#%%
len(np.subtract([1,23], [1,30]))

#%%
