#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
from bs4 import BeautifulSoup
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import json 
from geopy.geocoders import Nominatim

import geocoder
import requests
from pandas.io.json import json_normalize 
import matplotlib.cm as cm
import matplotlib.colors as colors

from sklearn.cluster import KMeans 
import folium 
import wget as wget


# In[2]:


result = requests.get("https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M")
src=result.content
soup= BeautifulSoup(src,'html.parser')
table = soup.find('table',{'wikitable sortable'})


# In[3]:


rows = table.find_all("tr")
l=[]
for r in rows:
    td=r.find_all("td")
    rw=[r.text for r in td]
    l.append(rw)
Canada=pd.DataFrame(l,columns=["Postcode","Borough","Neighbourhood"])


# In[4]:


Canada


# In[5]:


Canada=Canada[Canada.Borough!='Not assigned']


# In[6]:


Canada['Neighbourhood']=Canada['Neighbourhood'].replace(to_replace='Not assigned\n',value=Canada['Borough'])


# In[7]:


Canada = Canada.groupby('Postcode').agg({'Borough':'first', 
                             'Neighbourhood': ', '.join }).reset_index()


# In[8]:


def get_details(postal):
    lat_long= None
    while(lat_long is None):
         geo=geocoder.arcgis('{},Toronto,Ontario'.format(postal))
         lat_long = geo.latlng
    return lat_long


# In[9]:


cod=Canada['Postcode']
Coordinate=[get_details(cod) for cod in cod.tolist()]

Cords=pd.DataFrame(Coordinate, columns=['Latitude','Longitude'])
Canada['Latitude']=Cords['Latitude']
Canada['Longitude']=Cords['Longitude']


# In[10]:


Canada.drop(['Postcode'],axis= 1,inplace=True)
Canada 


# In[11]:


with open('C:/Users/acer/.anaconda/navigator/nyu_2451_34572-geojson.json') as json_data:
                 newyork_data = json.load(json_data)


# In[12]:


neighborhoods_data = newyork_data['features']


# In[13]:


column_names = ['Borough', 'Neighborhood', 'Latitude', 'Longitude'] 

new_york = pd.DataFrame(columns=column_names)


# In[14]:


for data in neighborhoods_data:
    borough = neighborhood_name = data['properties']['borough'] 
    neighborhood_name = data['properties']['name']
        
    neighborhood_latlon = data['geometry']['coordinates']
    neighborhood_lat = neighborhood_latlon[1]
    neighborhood_lon = neighborhood_latlon[0]
    
    new_york = new_york.append({'Borough': borough,
                                          'Neighborhood': neighborhood_name,
                                          'Latitude': neighborhood_lat,
                                          'Longitude': neighborhood_lon}, ignore_index=True)


# In[22]:


new_york=new_york.iloc[:153,:]
new_york


# In[23]:


CLIENT_ID = 'H0ODNDLWAN12HG3ZXXGWMBLU0AV1SXIS0E0ILO4DL5ZNVPK1'
CLIENT_SECRET = 'GZGTBFWNX1Y1XMA4Z2P1KWJQLSFI24L1FNLU4FI0UJCCMRS0' 
VERSION = '20191204' 


# In[24]:


def getNearbyVenues(code,names, latitudes, longitudes, radius=500):
    
    venues_list=[]
    for codes,name, lat, lng in zip(code,names, latitudes, longitudes):
        
        url = 'https://api.foursquare.com/v2/venues/explore?ll={},{}&client_id={}&client_secret={}&v={}&radius={}&limit={}'.format(
            lat, 
            lng,
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            500, 
            40)
            
        results = requests.get(url).json()['response']['groups'][0]['items']
        
        venues_list.append([(
            codes,
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Borough','Neighborhood', 
                  'Neighborhood Latitude', 
                  'Neighborhood Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)


# In[18]:


canadian_venues = getNearbyVenues(code=Canada['Borough'],
                                    names=Canada['Neighbourhood'],
                                   latitudes=Canada['Latitude'],
                                   longitudes=Canada['Longitude']
                                  )
canadian_venues


# In[25]:


newyork_venues = getNearbyVenues( code=new_york['Borough'],
                                   names=new_york['Neighborhood'],
                                   latitudes=new_york['Latitude'],
                                   longitudes=new_york['Longitude']
                                  )
newyork_venues 


# In[26]:


canadian_venues.groupby(["Borough"]).count()


# In[28]:


newyork_venues.groupby(["Borough"]).count()


# In[29]:


canada_onehot = pd.get_dummies(canadian_venues[['Venue Category']], prefix="", prefix_sep="")

canada_onehot['Borough'] = canadian_venues['Borough'] 

fixed_columns = [canada_onehot.columns[-1]] + list(canada_onehot.columns[:-1])
canada_onehot = canada_onehot[fixed_columns]

canada_onehot.head()


# In[30]:


canada_grouped = canada_onehot.groupby('Borough').mean().reset_index()
canada_grouped


# In[31]:


ny_onehot = pd.get_dummies(newyork_venues[['Venue Category']], prefix="", prefix_sep="")

ny_onehot['Borough'] = newyork_venues['Borough'] 

fixed_columns = [ny_onehot.columns[-1]] + list(ny_onehot.columns[:-1])
ny_onehot = ny_onehot[fixed_columns]

ny_onehot.head()


# In[32]:


ny_grouped = ny_onehot.groupby('Borough').mean().reset_index()
ny_grouped


# In[33]:


num_top_venues = 3

for hood in canada_grouped['Borough']:
    print("----"+hood+"----")
    temp = canada_grouped[canada_grouped['Borough'] == hood].T.reset_index()
    temp.columns = ['venue','freq']
    temp = temp.iloc[1:]
    temp['freq'] = temp['freq'].astype(float)
    temp = temp.round({'freq': 2})
    print(temp.sort_values('freq', ascending=False).reset_index(drop=True).head(num_top_venues))
    print('\n')


# In[34]:


num_top_venues = 3

for hood in ny_grouped['Borough']:
    print("----"+hood+"----")
    temp = ny_grouped[ny_grouped['Borough'] == hood].T.reset_index()
    temp.columns = ['venue','freq']
    temp = temp.iloc[1:]
    temp['freq'] = temp['freq'].astype(float)
    temp = temp.round({'freq': 2})
    print(temp.sort_values('freq', ascending=False).reset_index(drop=True).head(num_top_venues))
    print('\n')


# In[35]:


def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]


# In[37]:


num_top_venues = 5

indicators = ['st', 'nd', 'rd']

columns = ['Borough']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

can_sorted = pd.DataFrame(columns=columns)
can_sorted['Borough'] = canada_grouped['Borough']

for ind in np.arange(canada_grouped.shape[0]):
    can_sorted.iloc[ind, 1:] = return_most_common_venues(canada_grouped.iloc[ind, :], num_top_venues)

can_sorted


# In[38]:


num_top_venues = 5

indicators = ['st', 'nd', 'rd']

columns = ['Borough']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

ny_sorted = pd.DataFrame(columns=columns)
ny_sorted['Borough'] = ny_grouped['Borough']

for ind in np.arange(ny_grouped.shape[0]):
    ny_sorted.iloc[ind, 1:] = return_most_common_venues(ny_grouped.iloc[ind, :], num_top_venues)

ny_sorted


# In[40]:


kclusters = 3

canada_clustering = canada_grouped.drop('Borough', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(canada_clustering)

kmeans.labels_[0:10] 


# In[41]:


can_sorted.insert(0, 'Cluster Labels', kmeans.labels_)

canada_merged = Canada

canada_merged = canada_merged.join(can_sorted.set_index('Borough'), on='Borough')


# In[42]:


canada_merged=canada_merged.dropna()
canada_merged.reset_index(inplace=True)


# In[43]:


canada_merged


# In[44]:


num_top_venues = 5

indicators = ['st', 'nd', 'rd']

columns = ['Borough']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

ny_sorted = pd.DataFrame(columns=columns)
ny_sorted['Borough'] = ny_grouped['Borough']

for ind in np.arange(ny_grouped.shape[0]):
    ny_sorted.iloc[ind, 1:] = return_most_common_venues(ny_grouped.iloc[ind, :], num_top_venues)

ny_sorted


# In[46]:


kclusters = 3

ny_clustering = ny_grouped.drop('Borough', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(ny_clustering)

kmeans.labels_[0:4] 


# In[47]:


ny_sorted.insert(0, 'Cluster Labels', kmeans.labels_)

ny_merged = new_york

ny_merged = ny_merged.join(ny_sorted.set_index('Borough'), on='Borough')


# In[48]:


ny_merged=ny_merged.dropna()
ny_merged.reset_index(inplace=True)


# In[49]:


ny_merged


# In[53]:


address = 'Toronto'

geolocator = Nominatim(user_agent="can_explorer")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude

canada_merged['Cluster Labels']=canada_merged['Cluster Labels'].astype(int)
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=10.5)

x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

markers_colors = []
for lat, lon, poi, cluster in zip(canada_merged['Latitude'], canada_merged['Longitude'], canada_merged['Borough'], canada_merged['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters


# In[56]:


address = 'New York'

geolocator = Nominatim(user_agent="ny_explorer")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude

ny_merged['Cluster Labels']=ny_merged['Cluster Labels'].astype(int)
ma_clusters = folium.Map(location=[latitude, longitude], zoom_start=10.5)

x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

markers_colors = []
for lat, lon, poi, cluster in zip(ny_merged['Latitude'], ny_merged['Longitude'], ny_merged['Borough'], ny_merged['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(ma_clusters)
       
ma_clusters


# # End of Capstone Project
