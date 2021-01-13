#!/usr/bin/env python
# coding: utf-8

# ## Methodology - Data Preperation

# In[5]:


import pandas as pd
import numpy as np


# Converted in to data frame

# In[2]:


pip install geopandas


# In[3]:


pip install geocoder


# In[4]:


get_ipython().system('conda install -c conda-forge geopy --yes ')
from geopy.geocoders import Nominatim 
import requests 


# ### 1. Hong Kong Neighbourhood Data

# In[6]:


ldf = pd.read_excel('/home/reika/Desktop/HKNeighbourhoods.xlsx')
ldf


# Getting data in to format of address for geocoder

# In[7]:


N = ldf.loc[0,'Neighbourhood']
C = 'HongKong'
address = N + ', ' + C
address


# In[8]:


geolocator = Nominatim(user_agent = 'hk_explorer')


# In[9]:


location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinates are {}, {}.'.format(latitude, longitude))


# In[10]:


ldf['Long'] = ""
ldf['Lat'] = ""


# In[15]:


x = range(len(ldf))
for n in x:
    N = ldf.loc[n,'Neighbourhood']
    C = 'HongKong'
    address = N + ', ' + C
    location = geolocator.geocode(address)
    if location:
        latitude = location.latitude
        longitude = location.longitude
    else:
        latitude = np.NaN
        longitude = np.NaN
    ldf.loc[n,'Long'] = longitude
    ldf.loc[n,'Lat'] = latitude


# In[16]:


ldf


# Dealing with NaN Values

# In[17]:


ldf['Long'].isnull().sum()


# In[18]:


ldf.loc[pd.isna(ldf["Long"]), :].index


# In[19]:


ldf.loc[79,'Neighbourhood'], ldf.loc[108,'Neighbourhood']


# Kiu Tsui Chau is also known as Sharp Island, which seems to come up with coordinates

# In[20]:


ldf.loc[79,'Neighbourhood'] = 'Sharp Island'


# In[21]:


Address1 = 'Sharp Island, HongKong'


# In[22]:


location = geolocator.geocode(Address1)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinates are {}, {}.'.format(latitude, longitude))


# In[23]:


ldf.loc[79,'Lat'] = latitude
ldf.loc[79,'Long'] = longitude


# In[24]:


ldf['Long'].isnull().sum()


# Dropping information other than the Neighbourhood Name and cooridnates which will be used later

# In[25]:


ldf.drop(['Country','Area', 'District'], axis = 1, inplace = True)
ldf


# ### 2. Hong Kong Covid Data (from last 14 days)

# In[26]:


cdf = pd.read_csv ('/home/reika/Downloads/building_list_eng.csv')
cdf


# In[27]:


cdf.drop(['Last date of residence of the case(s)', 'District'], axis = 1, inplace = True)
cdf


# Have to change case number in to count of cases

# In[28]:


cdf.loc[1247,'Related probable/confirmed cases']


# In[29]:


cdf.loc[1247,'Related probable/confirmed cases'].count(',')


# Actual number of cases is 7, therefore need to add one to each count of ','

# In[30]:


cdf['CaseCount'] = ""


# In[31]:


x = range(len(cdf))
for n in x:
    cc = cdf.loc[n,'Related probable/confirmed cases'].count(',')
    cdf.loc[n,'CaseCount'] = cc + 1


# In[32]:


cdf


# In[33]:


cdf.drop(['Related probable/confirmed cases'], axis = 1, inplace = True)
cdf


# Normalise

# In[34]:


from sklearn import preprocessing


# In[35]:


import numpy as np


# In[36]:


x = cdf['CaseCount'].values
x.shape


# In[37]:


x = x.reshape(-1, 1)
x.shape


# Normalization = all becomes 1, not good

# In[38]:


n_x = preprocessing.normalize(x)
ndf = pd.DataFrame(n_x)
ndf[0].value_counts()


# MinMax Scaler alot = 0 not good

# In[39]:


min_max_scaler = preprocessing.MinMaxScaler()
x_n = min_max_scaler.fit_transform(x)
ncdf = pd.DataFrame(x_n)


# In[40]:


ncdf


# In[41]:


ncdf[0].value_counts()


# MaxabsScaler

# In[42]:


from sklearn.preprocessing import MaxAbsScaler


# In[43]:


transformer = MaxAbsScaler().fit(x)
transformer


# In[44]:


X = transformer.transform(x)


# In[45]:


n_ccdf = pd.DataFrame(X)
n_ccdf


# In[46]:


n_ccdf[0].value_counts()


# Better as not = 0 when there is 1 case. Still want there to be come weighting if there is 1 case. Therefore will use MaxAbsScaler for normalization

# In[47]:


ccdf = pd.concat([cdf, n_ccdf], axis = 1, sort = False)
ccdf


# In[48]:


ccdf.drop(['CaseCount'], axis = 1, inplace = True)


# In[49]:


ccdf = pd.DataFrame(ccdf)


# In[50]:


ccdf.columns = ['Building Address', 'CaseCount']
ccdf


# Finding lat and long of cases

# In[59]:


import csv
import urllib.parse


# In[115]:


row_list = ['BuildingAddress', 'Lat', 'Long']
with open('data.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(row_list)


# In[116]:


testdf = pd.read_csv ('data.csv')
testdf


# In[114]:


x= range(3)
for n in x:
    address = ccdf.loc[n,'Building Address']
    url = 'https://nominatim.openstreetmap.org/search/' + urllib.parse.quote(address) +'?format=json'
    response = requests.get(url).json()
    if response:
        latitude = response[0]["lat"]
        longitude = response[0]["lon"]
    else:
        latitude = np.NaN
        longitude = np.NaN
    with open('data.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        row= [str(ccdf.loc[n,'Building Address']), str(latitude), str(longitude)]
        writer.writerow(row)


# In[62]:


testdf1 = pd.read_csv ('data.csv')
testdf1


# In[118]:


row_list = ['BuildingAddress', 'Lat', 'Long']
with open('data.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(row_list)


# In[119]:


testdf = pd.read_csv ('data.csv')
testdf


# In[120]:


x= range(len(ccdf))
for n in x:
    address = ccdf.loc[n,'Building Address']
    url = 'https://nominatim.openstreetmap.org/search/' + urllib.parse.quote(address) +'?format=json'
    print (n, end = "\r")
    response = requests.get(url).json()
    if response:
        latitude = response[0]["lat"]
        longitude = response[0]["lon"]
    else:
        latitude = np.NaN
        longitude = np.NaN
    with open('data.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        row= [str(ccdf.loc[n,'Building Address']), str(latitude), str(longitude)]
        writer.writerow(row)


# In[121]:


clld = pd.read_csv ('data.csv')
clld


# In[122]:


clld['Long'].isnull().sum()


# In[123]:


N = clld.loc[pd.isna(clld["Long"]), :].index
N


# In[128]:


clld[clld['BuildingAddress'].str.contains("(non-residential)")]


# In[129]:


NR = clld.loc[clld['BuildingAddress'].str.contains("(non-residential)"), :].index
NR


# In[130]:


Z = [i for i in N if i in NR]
len(Z)


# Not all the addresses with "Non-Residential" in the address = NA, but a large majority did (462/479). Therefore need to delete this from the building address.

# In[131]:


test2 = clld.loc[1690,"BuildingAddress"]
test2 = test2.rstrip('(non-residential)')
test2


# In[132]:


x = NR


# In[133]:


for n in x:
    K = clld.loc[n,"BuildingAddress"]
    K = K.rstrip('(non-residential)')
    clld.loc[n,"BuildingAddress"] = K


# In[134]:


clld


# Test:

# In[136]:


RB = clld.loc[1690,'BuildingAddress']
address = RB + "," + "Hong Kong"
url = 'https://nominatim.openstreetmap.org/search/' + urllib.parse.quote(address) +'?format=json'
response = requests.get(url).json()
latitude = response[0]["lat"]
longitude = response[0]["lon"]
print (latitude)
print(longitude)


# Rerunning Function

# In[139]:


x= range(len(clld))

for n in x:
    RB = clld.loc[n,'BuildingAddress']
    address = RB + "," + "Hong Kong"
    url = 'https://nominatim.openstreetmap.org/search/' + urllib.parse.quote(address) +'?format=json'
    print (n, end = "\r")
    response = requests.get(url).json()
    if response:
        latitude = response[0]["lat"]
        longitude = response[0]["lon"]
    else:
        latitude = np.NaN
        longitude = np.NaN
    with open('data1.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        row= [str(ccdf.loc[n,'Building Address']), str(latitude), str(longitude)]
        writer.writerow(row)


# In[140]:


clld1 = pd.read_csv ('data1.csv')
clld1


# In[142]:


clld1['22.4453931'].isnull().sum()


# ### 3. FourSquare Data

# In[52]:


CLIENT_ID = 'J24FWPJAKX21U4BYVUGR5E1JM1042EPCW1SFNDJEOA1QLRZK' # your Foursquare ID
CLIENT_SECRET = 'CNCJWJRDLRERYXZXCTF2Q3CRDAEPQJ5ELK2JDK3NXGMESHH0' # your Foursquare Secret
VERSION = '20180605' # Foursquare API version
LIMIT = 100 # A default Foursquare API limit value

print('MY credentails:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)


# In[53]:


def getNearbyVenues(names, latitudes, longitudes, radius=500):
    
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
        print(name)
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        # make the GET request
        results = requests.get(url).json()["response"]['groups'][0]['items']
        
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighborhood', 
                  'Neighborhood Latitude', 
                  'Neighborhood Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)


# In[54]:


HongKong_venues = getNearbyVenues(names=ldf['Neighbourhood'],
                                   latitudes=ldf['Lat'],
                                   longitudes=ldf['Long'])


# In[55]:


print(HongKong_venues.shape)
HongKong_venues.head()


# In[56]:


HongKong_venues.groupby('Neighborhood').count()


# In[57]:


print('There are {} uniques categories.'.format(len(HongKong_venues['Venue Category'].unique())))


# In[85]:


HK_onehot = pd.get_dummies(HongKong_venues[['Venue Category']], prefix="", prefix_sep="")
HK_onehot['Neighborhood'] = HongKong_venues['Neighborhood'] 

HK_onehot.head()


# In[86]:


Inedx_Num = HK_onehot.columns.get_loc('Neighborhood')
Inedx_Num


# In[87]:


fixed_columns = [HK_onehot.columns[164]] + list(HK_onehot.columns[:164]) + list(HK_onehot.columns[165:])
HK_onehot = HK_onehot[fixed_columns]
HK_onehot.head()


# In[88]:


HK_onehot.shape


# Grouping by neighbourhood and taking the mean of frequency of occurance of each category

# In[89]:


HK_grouped = HK_onehot.groupby('Neighborhood').mean().reset_index()
HK_grouped


# In[90]:


HK_grouped.shape


# Top 10 most common ventues in each neighbourhood

# In[92]:


def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]


# In[95]:


num_top_venues = 10

indicators = ['st', 'nd', 'rd']

columns = ['Neighborhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

nvs = pd.DataFrame(columns=columns)
nvs['Neighborhood'] = HK_grouped['Neighborhood']

for ind in np.arange(HK_grouped.shape[0]):
    nvs.iloc[ind, 1:] = return_most_common_venues(HK_grouped.iloc[ind, :], num_top_venues)

nvs


# In[ ]:




