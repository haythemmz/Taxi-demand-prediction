#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
#%%
df_jan_2015=pd.read_pickle("df_jan_2015.pkl")

#%%
df_feb_2015=pd.read_pickle("df_feb_2015.pkl")

#%%
df_mar_2015=pd.read_pickle("df_mar_2015.pkl")
#%%
df_jan_2015.shape
#%%
df_jan_2016=pd.read_pickle("df_jan_2016.pkl")
#%%

df_feb_2016=pd.read_pickle("df_feb_2016.pkl")

#%%

df_feb_2016=pd.read_pickle("df_feb_2016.pkl")
#%%
df_mar_2016=pd.read_pickle("df_mar_2016.pkl")
#%%
regions_list=list(df_jan_2015['pickup_region'].unique()) #%% list des region fichier de configuration 


#%%
def zero_demand(data,t1,t2,t3,regions_list):
    
    o_1=(np.datetime64(t3) - np.datetime64(t1)) / np.timedelta64(1, 's')
    o_2=(np.datetime64(t2) - np.datetime64(t1)) / np.timedelta64(1, 's')
    u_1=int(o_1/600)
    u_2=int(o_2/600)
    print(u_1,u_2)
    gr=data[['pickup_region', "time_bin", "trip_distance"]].groupby(by = ['pickup_region', "time_bin"]).count()
    p=gr.reset_index()
    ri=[]
    ti=[]
    de=[]
    for i in regions_list:
        for j in range(u_2,u_1):
            ri.append(i)
            ti.append(j)
            if j in list(p[p['pickup_region']==i]["time_bin"]):
                f=p.loc[(p['pickup_region']==i) & (p["time_bin"]==j)]["trip_distance"].values[0]
                de.append(f)
            
            else:
                de.append(0)
    df=pd.DataFrame(list(zip(ri,ti,de)), columns=['regions','time_bin','taxi_demand'])
    return df 
#%%
t1='2015-01-01T00:00:00Z'
t2='2015-02-01T00:00:00Z'
t3='2015-03-01T00:00:00Z'
df_feb_2015_demand= zero_demand(data=df_feb_2015,t1=t1,t2=t2,t3=t3, regions_list=regions_list)

#%%
t1='2015-01-01T00:00:00Z'
t2='2015-03-01T00:00:00Z'
t3='2015-04-01T00:00:00Z'
df_mar_2015_demand= zero_demand(data=df_mar_2015,t1=t1,t2=t2,t3=t3, regions_list=regions_list)

#%%
t1='2016-01-01T00:00:00Z'
t2='2016-01-01T00:00:00Z'
t3='2016-02-01T00:00:00Z'
df_jan_2016_demand= zero_demand(data=df_jan_2016,t1=t1,t2=t2,t3=t3, regions_list=regions_list)

#%%
print(df_jan_2016['time_bin'].min())
print(df_jan_2016['time_bin'].max())

#%%
t1='2016-01-01T00:00:00Z'
t2='2016-02-01T00:00:00Z'
t3='2016-03-01T00:00:00Z'
df_feb_2016_demand= zero_demand(data=df_feb_2016,t1=t1,t2=t2,t3=t3, regions_list=regions_list)

#%%
t1='2016-01-01T00:00:00Z'
t2='2016-03-01T00:00:00Z'
t3='2016-04-01T00:00:00Z'
df_mar_2016_demand= zero_demand(data=df_mar_2016,t1=t1,t2=t2,t3=t3, regions_list=regions_list)

#%%
df_mar_2015_demand.head(10)
#%%
print(df_feb_2016_demand['time_bin'].min())
print(df_feb_2015_demand['time_bin'].min())
print(df_feb_2016_demand['time_bin'].max())
print(df_feb_2015_demand['time_bin'].max())

#%%
df_feb_demand['time_bin'].min()-df_mar_2015_demand['time_bin'].min()

#%%
df_feb_demand['time_bin'].max()-df_mar_2015_demand['time_bin'].max()
#%%
df_feb_2015_demand=df_feb_demand

#%%
t1='2015-01-01T00:00:00Z'
t2='2015-01-01T00:00:00Z'
t3='2015-02-01T00:00:00Z'
df_jan_2015_demand= zero_demand(data=df_jan_2015,t1=t1,t2=t2,t3=t3, regions_list=regions_list)

#%%
print(df_jan_2015_demand['time_bin'].min())
print(df_jan_2015_demand['time_bin'].min())
print(df_jan_2016_demand['time_bin'].max())
print(df_jan_2015_demand['time_bin'].max())

#%%
print(df_feb_2016_demand['time_bin'].min())
print(df_feb_2015_demand['time_bin'].min())
print(df_feb_2016_demand['time_bin'].max())
print(df_feb_2015_demand['time_bin'].max())

# problem of feb 29 in 2016 
#%%
print(df_mar_2016_demand['time_bin'].min())
print(df_mar_2015_demand['time_bin'].min())
print(df_mar_2016_demand['time_bin'].max())
print(df_mar_2015_demand['time_bin'].max())

#%%
df_feb_2016_demand=df_feb_2016_demand[df_feb_2016_demand['time_bin']<8496]

#%%
df_mar_2016_demand['time_bin']=df_mar_2016_demand['time_bin'].apply(lambda x : x-144)

#%%
def plot_demand_forgiven_region(region):

    plt.figure(figsize=(10,4))
    plt.plot(df_jan_2016_demand[df_jan_2016_demand['regions']==region].set_index('time_bin')['taxi_demand'], color ='limegreen')
    plt.plot(df_feb_2016_demand[df_feb_2016_demand['regions']==region].set_index('time_bin')['taxi_demand'], color ='mintcream')
    plt.plot(df_mar_2016_demand[df_mar_2016_demand['regions']==region].set_index('time_bin')['taxi_demand'], color ='cyan')

    plt.figure(figsize=(10,4))
    plt.plot(df_jan_2015_demand[df_jan_2015_demand['regions']==region].set_index('time_bin')['taxi_demand'], color ='limegreen')
    plt.plot(df_feb_2015_demand[df_feb_2015_demand['regions']==region].set_index('time_bin')['taxi_demand'], color ='mintcream')
    plt.plot(df_mar_2015_demand[df_mar_2015_demand['regions']==region].set_index('time_bin')['taxi_demand'], color ='cyan')
#%%

plot_demand_forgiven_region(5)




#%%

pickle.dump( df_jan_2015_demand, open( "df_jan_2015_demand.pkl", "wb" ) )
pickle.dump( df_feb_2015_demand, open( "df_feb_2015_demand.pkl", "wb" ) )
pickle.dump(df_mar_2015_demand, open( "df_mar_2015_demand.pkl", "wb" ) )


#%%
pickle.dump( df_jan_2016_demand, open( "df_jan_2016_demand.pkl", "wb" ) )
pickle.dump( df_feb_2016_demand, open( "df_feb_2016_demand.pkl", "wb" ) )
pickle.dump(df_mar_2016_demand, open( "df_mar_2016_demand.pkl", "wb" ) )

#%%
df_mar_2016_demand['taxi_demand'].equals(df_mar_2015_demand['taxi_demand'])