
#%%
import pandas as pd 
import dask.dataframe as dd
import folium
import matplotlib.pyplot as plt
from datetime import datetime 
import time 
import numpy as np
import seaborn as sns 
import pickle 
from sklearn.cluster import MiniBatchKMeans, KMeans
import gpxpy.geo

#%%
jan_2015=dd.read_csv('2015\yellow_tripdata_2015-01.csv')
feb_2015=dd.read_csv('2015\yellow_tripdata_2015-02.csv')
mar_2015=dd.read_csv('2015\yellow_tripdata_2015-03.csv')
#%%
jan_2016=dd.read_csv('2016\yellow_tripdata_2016-01.csv')
feb_2016=dd.read_csv('2016\yellow_tripdata_2016-02.csv')
mar_2016=dd.read_csv('2016\yellow_tripdata_2016-03.csv')


#%%
def print_outside_NewYork(df, number_of_outsides):
    out_NewYork = df[(df['pickup_latitude'] <= 40.5774) or  (df['pickup_longitude'] <= -74.15) or(df['pickup_latitude'] >= 40.9176) or (df['pickup_longitude'] >= -73.7004)]
    NewYork_Map = folium.Map(location = [40.5774, -73.7004], tiles = "Stamen Toner")
    first_out-outside_pickups = out_NewYork.head(number_of_outsides)
    for i,j in first_out-outside_pickups.iterrows():
        if j["pickup_latitude"] != 0:
            folium.Marker([j["pickup_latitude"], j["pickup_longitude"]]).add_to(NewYork_Map)
    return NewYork_Map

#%%
df=jan_2017
out_NewYork = df[(df['pickup_latitude'] <= 40.5774) | (df['pickup_longitude'] <= -74.15) | (df['pickup_latitude'] >= 40.9176) | (df['pickup_longitude'] >= -73.7004)]
NewYork_Map = folium.Map(location = [40.5774, -73.7004], tiles = "Stamen Toner")
first_out_outside_pickups = out_NewYork.head(100)
for i,j in first_out_outside_pickups.iterrows():
    if j["pickup_latitude"] != 0:
        folium.Marker([j["pickup_latitude"], j["pickup_longitude"]]).add_to(NewYork_Map)
NewYork_Map
#%%
oct_2015.dtypes



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
    a=(np.array(end)-np.array(start))/float(60)
    duration=list(map(lambda x : x.total_seconds() , a))
    
    df_new=df[['trip_distance','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','store_and_fwd_flag','payment_type','fare_amount','extra','mta_tax','tip_amount','tolls_amount','improvement_surcharge']].compute()
    df_new['duration']=duration
    df_new['pickup_time']=start
    df_new['avg_speed']=(df_new['trip_distance']*60)/df_new['duration']
    return df_new

#%%
df_jan_2015=time_construction(jan_2015)



#%%

data=time_construction(jan_2017)

#%%
pickle.dump( df_oct_2015, open( "df_oct_2015.pkl", "wb" ) )
#%%
sns.boxplot(y='duration',data=data)

#%%
def percentilles(df,l, col):
    for j in l:
        print("{} percentille = {}".format(j,df[col].quantile(j)))
#%%
l=list(np.linspace(0,1,10))
percentilles(df=df,l=l,col='duration')

#%%
l=list(np.linspace(0,0.01,10))
percentilles(df=df,l=l,col='duration')
#%%
duration_downbound=0
#%%
l=list(np.linspace(0.9,1,10))
percentilles(df=df,l=l,col='duration')
#%%
l=list(np.linspace(0.999,1,10))
percentilles(df=df,l=l,col='duration')
#%%
duration_upbound=720
#%%
print(data.shape) # save shape of global outliers comparasion
data=data[(data['duration'] >0) & (data['duration'] <720) ] #12 hours
print(data.shape)
#%%
sns.boxplot(y='trip_distance',data=data)

#%%
l=list(np.linspace(0.99,1,10))
percentilles(df=data,l=l,col='trip_distance')

#%%
print(data.shape)
data=data[data['trip_distance'] <23 ] 
print(data.shape)

#%%
sns.boxplot(y='avg_speed',data=data)

#%%
l=list(np.linspace(0,1,10))
percentilles(df=data,l=l,col='avg_speed')

#%%
print(data.shape)
data=data[(data['avg_speed'] >0) & (data['avg_speed'] <50) ]
print(data.shape)

#%%
sns.boxplot(y='fare_amount',data=data)

#%%
l=list(np.linspace(0,1,10))
percentilles(df=data,l=l,col='fare_amount')

#%%
l=list(np.linspace(0.99,1,10))
percentilles(df=data,l=l,col='fare_amount')

#%%
print(data.shape)
data=data[(data['fare_amount'] >0) & (data['fare_amount'] <53) ]
print(data.shape)

#%%
print(data.shape)
data=data[(data['pickup_latitude'] > 40.5774) &  (data['pickup_longitude'] > -74.15) & (data['pickup_latitude'] < 40.9176) & (data['pickup_longitude'] < -73.7004)]
print(data.shape)

#%%
print(data.keys())

#%%
def makingRegions(number_clusters):
    clusters = MiniBatchKMeans(n_clusters = number_clusters, batch_size = 10000).fit(coord)
    clustersCenters = clusters.cluster_centers_ 
    totalClusters = len(clustersCenters)
    return clustersCenters, totalClusters

#%%
def min_distance(regionCenters, totalClusters):
    good_points = 0
    bad_points = 0
    less_dist = []
    more_dist = []
    min_distance = 100000  #any big number can be given here
    for i in range(totalClusters):
        good_points = 0
        bad_points = 0
        for j in range(totalClusters):
            if j != i:
                distance = gpxpy.geo.haversine_distance(latitude_1 = regionCenters[i][0], longitude_1 = regionCenters[i][1], latitude_2 = regionCenters[j][0], longitude_2 = regionCenters[j][1])
                #you can check the documentation of above "gpxpy.geo.haversine_distance" at "https://github.com/tkrajina/gpxpy/blob/master/gpxpy/geo.py"
                #"gpxpy.geo.haversine_distance" gives distance between two latitudes and longitudes in meters. So, we have to convert it into miles.
                distance = distance/(1.60934*1000)   #distance from meters to miles
                min_distance = min(min_distance, distance) #it will return minimum of "min_distance, distance".
                if distance < 2:
                    good_points += 1
                else:
                    bad_points += 1
        less_dist.append(good_points)
        more_dist.append(bad_points)
    print("On choosing a cluster size of {}".format(totalClusters))
    print("Avg. Number clusters within vicinity where inter cluster distance < 2 miles is {}".format(np.ceil(sum(less_dist)/len(less_dist))))
    print("Avg. Number clusters outside of vicinity where inter cluster distance > 2 miles is {}".format(np.ceil(sum(more_dist)/len(more_dist))))
    print("Minimum distance between any two clusters = {}".format(min_distance))
    print("-"*10)
#%%
coord = df_oct_2015[["pickup_latitude", "pickup_longitude"]].values
neighbors = []
startTime = datetime.now()
for i in range(10, 100, 10):
    regionCenters, totalClusters = makingRegions(i)
    min_distance(regionCenters, totalClusters)
print("Time taken = "+str(datetime.now() - startTime))
# interclass distance < 2
# intera class distance > 0.5
#%%
coord = df_jan_2015[["pickup_latitude", "pickup_longitude"]].values
regions = MiniBatchKMeans(n_clusters = 40, batch_size = 10000).fit(coord)
df_jan_2015["pickup_region"] = regions.predict(df_jan_2015[["pickup_latitude", "pickup_longitude"]].values)
# convert to function and predict pickups for for each dataframe 

#%%
df_jan_2015["pickup_region"].unique()
#%%
data.dtypes

#%%
a=list(data["pickup_time"])

#%%
def time_parts (b,month,year):
    time_string=str(year)+'-'+str(month)+'-01T00:00:00Z'
    k=(b - np.datetime64(time_string)) / np.timedelta64(1, 's')
    return int((k/60)//10)

#%%
b=a[0]
month='01'
year='2015'
u=time_parts(b,month,year)
print(u)
print(b)

#%%
def time_bin(df,month,year):
    df['time_bin']=[time_parts(x,month,year) for x in df["pickup_time"]]
    return df

#%%
df=time_bin(data,month,year)
#%%
data.dtypes
#%%
def number_pickups(df):
    df['size']=df.groupby(['pickup_region','time_bin']).size().astype(int).reset_index(name='size')['size']
    return df 

#%%
def data_cleansing(df):
    #a=df.shape
    df=df[(df['pickup_latitude'] > 40.5774) &  (df['pickup_longitude'] > -74.15) & (df['pickup_latitude'] < 40.9176) & (df['pickup_longitude'] < -73.7004)]
    df=df[(df['duration'] >0) & (df['duration'] <720) ] #
    df=df[(df['trip_distance'] >0) & (df['trip_distance'] <23) ] #
    df=df[(df['avg_speed'] >0) & (df['avg_speed'] <50) ] #
    df=df[(df['fare_amount'] >0) & (df['fare_amount'] <53) ] #
    #b=df.shape
    #print(a[0]-b[0])
    return df 

#%%

df_oct_2015=time_construction(oct_2015)

#%%
df_nov_2015=time_construction(nov_2015)
#%%
df_nov_2015=data_cleansing(df_nov_2015)


#%%
df_nov_2015=time_construction(nov_2015)
#%%
df_nov_2015=data_cleansing(df_nov_2015)

#%%
df_oct_2015["pickup_region"] = regions.predict(df_oct_2015[["pickup_latitude", "pickup_longitude"]])
#%%
month='10'
year='2015'
df_nov_2015=time_bin(df_nov_2015,month,year)
#%%
df_oct_2015=data_cleansing(df_oct_2015)

#%%

df_oct_2015["pickup_region"] = regions.predict(df_oct_2015[["pickup_latitude", "pickup_longitude"]])

#%%
month='10'
year='2015'
df_oct_2015=time_bin(df_nov_2015,month,year)
#%%
df_oct_2015.head()
#%%
dec_2015
#%%
df_oct_2015=number_pickups(df_oct_2015)

#%%
# the block of code 
def contruct_frame(df,month,year,regions=regions):
    df=time_construction(df)
    df=data_cleansing(df)
    df["pickup_region"] = regions.predict(df[["pickup_latitude", "pickup_longitude"]])
    
    df=time_bin(df,month,year)
    df=number_pickups(df)
    return df

#%%
df_nov_2015=time_construction(nov_2015)
df_nov_2015=data_cleansing(df_nov_2015)
df_nov_2015["pickup_region"] = regions.predict(df_nov_2015[["pickup_latitude", "pickup_longitude"]])
month='10'
year='2015'
df_nov_2015=time_bin(df_nov_2015,month,year)
df_nov_2015=number_pickups(df_nov_2015)
#%%
df_jan_2015=contruct_frame(df=jan_2015,month='01',year='2015',regions=regions)
#%%
pickle.dump( df_jan_2015, open( "df_jan_2015.pkl", "wb" ) )
#%%
df_feb_2015=contruct_frame(df=feb_2015,month='01',year='2015',regions=regions)
#%%
pickle.dump( df_feb_2015, open( "df_feb_2015.pkl", "wb" ) )

#%%
df_mar_2015=contruct_frame(df=mar_2015,month='01',year='2015',regions=regions)

#%%
pickle.dump( df_mar_2015, open( "df_mar_2015.pkl", "wb" ) )

#%%
df_jan_2016=contruct_frame(df=jan_2016,month='01',year='2016',regions=regions)


#%%

pickle.dump( df_jan_2016, open( "df_jan_2016.pkl", "wb" ) )


#%%
df_feb_2016=contruct_frame(df=feb_2016,month='01',year='2016',regions=regions)

#%%
pickle.dump( df_feb_2016, open( "df_jan_2016.pkl", "wb" ) )

#%%
df_mar_2016=contruct_frame(df=mar_2016,month='01',year='2016',regions=regions)


#%%
pickle.dump( df_mar_2016, open( "df_mar_2016.pkl", "wb" ) )
#%%
df_feb_2016=contruct_frame(df=feb_2016,month='01',year='2016',regions=regions)

#%%
pickle.dump( df_feb_2016, open( "df_feb_2016.pkl", "wb" ) )