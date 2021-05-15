#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing libraries
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler
from scipy import linalg
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt 
import seaborn as sns 


# In[2]:


#reading csv
df = pd.read_csv("Mumbai1.csv")


# In[3]:


df


# In[4]:


#dropping unnecessary columns
df2 = df.drop(["Children's Play Area",'Intercom','Landscaped Gardens','Indoor Games','Maintenance Staff','24x7 Security'],axis = 'columns')
df2


# In[5]:


#cleaning
df2.isnull().sum()


# In[6]:


#checking "Area" column
df2['Area'].unique()


# In[7]:


df2['Price'].unique()


# In[8]:


#checking "New/Resale" column
df2['New/Resale'].unique()


# In[9]:


#checking "No. of Bedrooms" column
df2['No. of Bedrooms'].unique()


# In[10]:


#replacing 'no. of bedrooms' by 'bhk'
df2['bhk'] = df2['No. of Bedrooms'] 


# In[11]:


df3 = df2.drop(['No. of Bedrooms'],axis = 'columns')


# In[12]:


df3


# In[13]:


#defining a function that will help in finding float values in a column
def find_float(x):
    try:
        float(x)
    except:
        return False
    return True


# In[14]:


df3[~df3['Area'].apply(find_float)]


# In[15]:


df3[~df3['Price'].apply(find_float)]


# In[16]:


#feature eng
#calculating price per area
df4 = df3.copy()
df4['price_per_sqft'] = df4['Price']/df4['Area']
df4


# In[17]:


#checking uniques in 'location'
df4.Location.unique()


# In[18]:


len(df4.Location.unique())


# In[19]:


#checking entries per location
df4.Location = df4.Location.apply(lambda x: x.strip())
location_stats = df4.groupby('Location')['Location'].agg('count').sort_values(ascending=False)
location_stats


# In[20]:


location_stats_other = location_stats[location_stats<=10]
location_stats_other


# In[21]:


len(location_stats_other)


# In[22]:


df5 = df4.copy()


# In[23]:


df5.Location = df5.Location.apply(lambda x: 'other' if x in location_stats_other else x)


# In[24]:


df5


# In[25]:


len(df5.Location.unique())


# In[26]:


#remove outliers
#assume 1bhk to occupy atleast 350sqft
df5[df5['Area']/(df5['bhk'])<350]


# In[27]:


#removing rows with area/bhk less than 350
df6 = df5[~(df5['Area']/(df5['bhk'])<350)]
df6


# In[28]:


df6.price_per_sqft.describe()


# In[29]:


#checking price per sqft for a location
df5[(df5['Location']=='Kharghar')]


# In[30]:


#defining a function to remove outliers
def remove_outliers_pps(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('Location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft>(m-st)) & (subdf.price_per_sqft<(m+st))]
        df_out = pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out


# In[31]:


df7 = remove_outliers_pps(df6)
df7


# In[32]:


df8 = df7.drop(['price_per_sqft'],axis='columns')
df8


# In[33]:


#one hot encoding/dummy
#creating a dummy df with locations as columns
dummies = pd.get_dummies(df8.Location)
dummies


# In[34]:


#concatinating dummies with our df
df9 = pd.concat([df8,dummies.drop('other',axis='columns')],axis='columns')
df9


# In[35]:


df9 = pd.concat([df8,dummies.drop('other',axis='columns')],axis='columns')
df9


# In[36]:


df10 = df9.drop('Location',axis='columns')
df10


# In[37]:


#renaming column names for convenience
df10 = df10.rename(columns={'New/Resale': 'New_or_Resale',
                            'Lift Available': 'Lift_Available',
                            'Car Parking': 'Car_Parking',
                            'Gas Connection': 'Gas_Connection',
                            'Jogging Track': 'Jogging_Track',
                            'Swimming Pool': 'Swimming_Pool'})
df10


# In[38]:


df10 = df10.drop('0',axis='columns')


# In[39]:


#final dataset
df10


# In[40]:


#training model
X = df10.drop('Price',axis='columns')
X


# In[41]:


y = df10.Price
y


# In[42]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=2)


# In[43]:


#checking score
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)
regressor.score(X_test,y_test)


# In[44]:


X_test[0:1]


# In[45]:


regressor.predict(X_test[0:1])


# In[46]:


def predict_price(Location,Area,New_or_Resale,Gymnasium,Lift_Available,Car_Parking,Clubhouse,Gas_Connection,Jogging_Track,Swimming_Pool,bhk):
    loc_index = np.where(X.columns==Location)[0][0]
    x = np.zeros(len(X.columns))
    x[0] = Area
    x[1] = New_or_Resale
    x[2] = Gymnasium
    x[3] = Lift_Available
    x[4] = Car_Parking
    x[5] = Clubhouse
    x[6] = Gas_Connection
    x[7] = Jogging_Track
    x[8] = Swimming_Pool
    x[9] = bhk
    if loc_index >= 0:
        x[loc_index] = 1
    
    return regressor.predict([x])[0]


# In[47]:


#defining a function to predict the price based on input from user
def predict_prices(Location,Area,New_or_Resale,Gymnasium,Lift_Available,Car_Parking,Clubhouse,Gas_Connection,Jogging_Track,Swimming_Pool,bhk):
    index=0
    for i in X.columns:
        if(X.columns[index]==Location):
            loc_index=index
            break
        index=index+1
    x = np.zeros(len(X.columns))

    x[0] = Area
    x[1] = New_or_Resale
    x[2] = Gymnasium
    x[3] = Lift_Available
    x[4] = Car_Parking
    x[5] = Clubhouse
    x[6] = Gas_Connection
    x[7] = Jogging_Track
    x[8] = Swimming_Pool
    x[9] = bhk
    if loc_index >= 0:
        x[loc_index] = 1
    return regressor.predict([x])[0]


# In[48]:


predict_price('Vashi',10000,0,1,0,1,1,0,1,1,2)


# In[49]:


predict_price('Kharghar',720,0,0,1,1,0,0,0,0,1)


# In[50]:


predict_prices('Kharghar',720,0,0,1,1,0,0,0,0,1)


# In[51]:


y_pred = regressor.predict(X_test)


# In[52]:


y_pred


# In[53]:


#using a histogram to check accuracy of our model
sns.distplot(y_pred,  kde=False, label='Prediction')
sns.distplot(y_test,  kde=False,label='Test')
plt.legend
plt.title('Compairing our prediction model with test model')
plt.xlabel('Area')
plt.ylabel('Price')


# In[54]:


#comparing test values and predicted values
plt.scatter(X_test['Area'],y_test, color = 'red')
plt.scatter(X_test['Area'],y_pred, color = 'blue')
plt.xlabel('Area')
plt.ylabel('Price')
plt.show()


# In[55]:


plt.scatter(X_train['Area'],y_train, color = 'red')
plt.plot(X_test['Area'],y_pred, color = 'blue')
plt.xlabel('Area')
plt.ylabel('Price')
plt.show()


# In[56]:


#saving our model using pickle
import pickle
with open('realestate_price_model_mumbai.pickle','wb') as f:
    pickle.dump(regressor,f)


# In[57]:


#saving name of columns in json
import json
columns = {
    'data_columns': [col.lower() for col in X.columns]
}
with open('locations.json','w') as f:
    f.write(json.dumps(columns))


# In[ ]:




