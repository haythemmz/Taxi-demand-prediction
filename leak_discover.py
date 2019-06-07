#%%
import pandas as pd 
import seaborn as sns 
import numpy as np 



#%%
df=pd.read_pickle("just_frame.pkl")

#%%
def time_construction(df):
    d_time=df[['tpep_dropoff_datetime','tpep_pickup_datetime']].compute()
    start=[convert_time(x) for x in d_time['tpep_pickup_datetime'].values]
    end=[convert_time(x) for x in d_time['tpep_dropoff_datetime'].values]
    a=(np.array(end)-np.array(start))/float(60)
    duration=list(map(lambda x : x.total_seconds() , a))
    
    df_new=df[['trip_distance','store_and_fwd_flag','payment_type','fare_amount','extra','mta_tax','tip_amount','tolls_amount','improvement_surcharge']].compute()
    df_new['duration']=duration
    df_new['pickup_time']=start
    df_new['avg_speed']=(df_new['trip_distance']*60)/df_new['duration']
    return df_new 

#%%
df.dtypes

#%%
sns.boxplot(y='duration',data=df)

#%%
def percentilles(df,l, col):
    for j in l:
        print("{} percentille = {}".format(j,df[col].quantile(j)))



#%%
l=list(np.linspace(0.999,1,10))
percentilles(df=df,l=l,col='duration')

#%%
sns.distplot(df['duration'],hist=False)

#%%
max(l)