
#%%
import pandas as pd 
import dask.dataframe as dd
import folium
import matplotlib.pyplot as plt
from datetime import datetime 
import time 
import numpy as np


#%%
march_2017=dd.read_csv("2017/yellow_tripdata_2017-03.csv")
jan_2017=dd.read_csv('2017/yellow_tripdata_2017-01.csv')
feb_2017=dd.read_csv('2017/yellow_tripdata_2017-02.csv')

#%%

jan_2017.head()

#%%
jan_2017.dtypes

#%%
plt.hist(list(jan_2017['PULocationID']))

#%%

new_york_location=[-74.15,-73.7004,40.5774,40.9176]
out_newyork=march_2017[march_2017[]]

new_york=folium.Map(location=[-40.73,-73.99], )







#%%

# Trip duration 

def convert_time(x):
    t=datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    return t

#%%
def time_construction(df):
    d_time=df[['tpep_dropoff_datetime','tpep_pickup_datetime']].compute()
    start=[convert_time(x) for x in d_time['tpep_pickup_datetime'].values]
    end=[convert_time(x) for x in d_time['tpep_dropoff_datetime'].values]
    duration=(np.array(end)-np.array(start))/float(60)
    df_new=[['trip_distance','RatecodeID','store_and_fwd_flag','PULocationID','DOLocationID','payment_type','fare_amount','extra','mta_tax','tip_amount','tolls_amount','improvement_surcharge']]
    df_new['duration']=list(duration)
    df_new['pickup_time']=start
    df_new['avg_speed']=(df_new['trip_distance']*60)/df_new['duration']
    return df_new

#%%
jan_2017_df=time_construction(jan_2017)

#%%
jan_2017.loc[0,'tpep_dropoff_datetime']


#%%
def drop_out_NewYork(df)