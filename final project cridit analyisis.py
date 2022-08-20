#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing Libraries 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.

import time, warnings
import datetime as dt

#visualizations
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import klib
warnings.filterwarnings("ignore")
#!pip install dataprep


# In[2]:


##impoting tha dataset
df=pd.read_excel("C:\\Users\shamanth\Downloads\CreditAnalysis_data.xlsx")


# In[3]:


df.head()


# In[4]:


# check the summary info of the dataframe
df.info()


# In[5]:


df.shape


# ## EDA PART using klib

# In[6]:


klib.corr_mat(df)


# In[7]:


plt.figure(figsize =(12,9))
s=sns.heatmap(df.corr(),
              annot = True,
              cmap = 'RdBu',
              vmin = -1,
              vmax = 1)
s.set_yticklabels(s.get_yticklabels(),rotation = 0, fontsize = 12)
s.set_xticklabels(s.get_xticklabels(),rotation = 90, fontsize = 12)
plt.title('correlation Heatmap')
plt.show()


# In[8]:


klib.corr_plot(df)


# In[9]:


from dataprep.eda import create_report


# In[10]:


create_report(df)


# In[11]:


klib.dist_plot(df)


# In[12]:


# check for missing values
df.isnull().sum()


# In[13]:


# select the portion of dataframe with no missing values for the column "ordereditem_product_id  "
df = df[pd.notnull(df["ordereditem_product_id"])]
df.isnull().sum()


# In[14]:


df.info()


# In[15]:


df.columns


# In[16]:


print("There are {} duplicated values.".format(df.duplicated().sum()))
df[df.duplicated(keep=False)].head(5)


# In[17]:


# converting "created" column to datetime
df['created'] = df['created'].astype('datetime64[ns]')
df.head()


# ### Data Insights

# Since the data is from an cridite analysis, we will look at where their order_id originate from i.e. 'group'

# In[18]:


# filter retailers by top 10 countries in percentage
df.group.value_counts(normalize=True)[:10]


# Hyderabad alone accounts for 88.4% of all groups
# more order id created in Hyderabad

# In[19]:


# visualize in bar chart
df.group.value_counts(normalize=True)[:10].plot(kind="bar")


# We can filter and only select groups from hyderabad, which are considered as "local customers" for the cridite analysis since the  retailer in hyderabad. This will serves as a good starting point for our analysis and also for the  retailer to focus their marketing effort. becouse see tha above graps hyderabad having more oreder ids. 

# In[20]:


## extracting unique value
print(df['created'].unique())
print(df['order_id'].unique())
print(df['order_status'].unique())
print(df['ordereditem_quantity'].unique())
print(df['prod_names'].unique())
print(df['ordereditem_unit_price_net'].unique())
print(df['value'].unique())
print(df['group'].unique())
print(df['dist_names'].unique())
print(df['retailer_names'].unique())
print(df['bill_amount'].unique())


# In[21]:


## Unique order ids and order share of top retailers
len(df.order_id.unique())


# In[22]:


## Unique retailernames and order share of top retailers
len(df.retailer_names.unique())


# In[23]:


## Unique item ordereditem_product_id and their description
#Find number of unique items in order_status

print(f"Number of unique item ordereditem_product_id: {len(df.ordereditem_product_id.unique())}")
print(f"Number of unique item order_status: {len(df.order_status.unique())}")


# ## Compute RFM value
# #Feature Engineering - Building features for RFM model:
# Recency: We fix a reference date for finding the recent transactions. The reference date would be a day after the most recent transaction date in the dataset. Then we calculate the days difference between the most recent transaction carried out by the retailer and this reference date
# 

# ## Calculating Recency

# In[24]:


# last date available in our dataset
df.created.max()


# In[25]:


# setting now to calculate time differences
now = dt.date(2018,12,3)


# In[26]:


# group by retailers by last date they purchased

recency_df = df.groupby(['retailer_names'],as_index=False)['created'].max()
recency_df.columns = ['retailer_names','LastPurchaseDate']
recency_df['LastPurchaseDate'] = pd.DatetimeIndex(recency_df.LastPurchaseDate).date
recency_df.head()


# In[27]:


# calculate how often the retailers are buying in the last few days

recency_df['Recency'] = recency_df.LastPurchaseDate.apply(lambda x : (now - x).days)

# dropping LastPurchase Date
recency_df.drop(columns=['LastPurchaseDate'],inplace=True)

# checking recency
recency_df.head()


# Now, let us examine how a single unit code has multiple order_status:

# ## Calculating Frequency
# We are here calculating the frequency of frequent transactions of the customer in ordering/buying some product from the company.

# In[28]:


# calculating frequency
frequency_df = df.copy()
frequency_df.drop_duplicates(subset=['retailer_names','created'], keep="first", inplace=True) 
frequency_df = frequency_df.groupby('retailer_names',as_index=False)['created'].count()
frequency_df.columns = ['retailer_names','Frequency']
frequency_df.head()


# ## Calculating Monetary Value
# Here we are calculating the monetary value of customer spend on purchasing products from the company.

# In[29]:


## Check summed up spend of customers

monetary_df=df.groupby('retailer_names',as_index=False)['value'].sum()
monetary_df.columns = ['retailer_names','Monetary']
monetary_df.head()


# In[30]:


# putting recency and frequency together
rf = recency_df.merge(frequency_df,left_on='retailer_names',right_on='retailer_names')

# combining with monetary values
rfm = rf.merge(monetary_df,left_on='retailer_names',right_on='retailer_names')

rfm.set_index('retailer_names',inplace=True)

# saving to file
rfm.to_csv('rfm.csv', index=False)

# checking the dataframe
rfm.head()


# In[31]:


# Descriptive Statistics (Recency)
rfm.Recency.describe()


# In[32]:


# Recency distribution plot
import seaborn as sns
x = rfm['Recency']

ax = sns.displot(x)


# In[33]:


# Descriptive Statistics (Frequency)
rfm.Frequency.describe()


# In[34]:


# Frequency  distribution plot
import seaborn as sns
x = rfm['Frequency']

ax = sns.displot(x)


# In[35]:


# Descriptive Statistics (Monetary)
rfm.Monetary.describe()


# In[36]:


# Monetary  distribution plot
import seaborn as sns
x = rfm['Monetary']

ax = sns.displot(x)


# In[37]:


#Split into four segments using quantiles
quantiles = rfm.quantile(q=[0.25,0.5,0.75])
quantiles = quantiles.to_dict()
quantiles


# In[38]:


#Functions to create R, F and M segments
def RScoring(x,p,d):
    if x <= d[p][0.25]:
        return 1
    elif x <= d[p][0.50]:
        return 2
    elif x <= d[p][0.75]: 
        return 3
    else:
        return 4

def FnMScoring(x,p,d):
    if x <= d[p][0.25]:
        return 4
    elif x <= d[p][0.50]:
        return 3
    elif x <= d[p][0.75]: 
        return 2
    else:
        return 1


# In[39]:


#Calculate Add R, F and M segment value columns in the existing dataset to show R, F and M segment values
rfm['R'] = rfm['Recency'].apply(RScoring, args=('Recency',quantiles,))
rfm['F'] = rfm['Frequency'].apply(FnMScoring, args=('Frequency',quantiles,))
rfm['M'] = rfm['Monetary'].apply(FnMScoring, args=('Monetary',quantiles,))
rfm.head()


# In[40]:


#Calculate and Add RFMGroup value column showing combined concatenated score of RFM
rfm['RFMGroup'] = rfm.R.map(str) + rfm.F.map(str) + rfm.M.map(str)

#Calculate and Add RFMScore value column showing total sum of RFMGroup values
rfm['RFMScore'] = rfm[['R', 'F', 'M']].sum(axis = 1)
rfm.head()


# In[41]:


#Assign Loyalty Level to each customer
Loyalty_Level = ['Platinum', 'Gold', 'Silver', 'Bronze']
Score_cuts = pd.qcut(rfm.RFMScore, q = 4, labels = Loyalty_Level)
rfm['RFM_Loyalty_Level'] = Score_cuts.values
rfm.reset_index().head()


# In[42]:


#Validate the data for RFMGroup = 111
rfm[rfm['RFMGroup']=='111'].sort_values('Monetary', ascending=False).reset_index().head(10)


# In[43]:


rfm.RFM_Loyalty_Level.value_counts()


# In[44]:


#Extracting Independent and dependent Variable  
x= rfm.iloc[:,:-1]  
y= rfm.iloc[:,-1]


# In[45]:


# Splitting the dataset into training and test set.  
from sklearn.model_selection import train_test_split  
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.20, random_state=42)


# In[46]:


rfm.columns


# In[47]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

ohe = OneHotEncoder()
ohe.fit(rfm[['Recency', 'Frequency', 'Monetary', 'R', 'F', 'M', 'RFMGroup', 'RFMScore']])


# In[48]:


column_trans = make_column_transformer((OneHotEncoder(categories=ohe.categories_), ['Recency', 'Frequency', 'Monetary', 'R', 'F', 'M', 'RFMGroup', 'RFMScore']), 
                                                    remainder='passthrough')


# In[49]:


#Fitting RandomForest classifier to the training set  
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
classifier = RandomForestClassifier(n_estimators= 100, criterion="gini")  


# In[50]:


rf_pipe = make_pipeline(column_trans, classifier)


# In[51]:


rf_pipe.fit(x_train, y_train)


# In[52]:


## Evaluation of Test data (Prediction)
y_pred = rf_pipe.predict(x_test)


# ## Creating confusion_matrix

# In[53]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
result = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(result)

result1 = classification_report(y_test, y_pred)
print("Classification Report:",)
print (result1)

result2 = accuracy_score(y_test,y_pred)
print("Accuracy:",result2)


# In[54]:


# Train Data Accuracy
accuracy_score(y_train, rf_pipe.predict(x_train))


# In[55]:


# saving the model
# importing the model
import pickle

rfr_model = 'rf_cl_model.pkl'
pickle.dump(rf_pipe, open('rfr_model', 'wb'))


# In[ ]:





# In[ ]:




