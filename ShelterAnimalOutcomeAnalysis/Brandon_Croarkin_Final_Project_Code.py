
# coding: utf-8

# # Shelter Animal Outcomes

# In[57]:


import pandas as pd
import numpy as np
import os
import seaborn as sns
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn import preprocessing


# In[58]:


os.chdir(r'C:\Users\brcro\OneDrive\Documents\Syracuse\IST 652 - Scripting for Data Analysis\Project')
#os.chdir(r'C:\Users\Brandon Croarkin\Documents\Education\Syracuse\Scripting')


# ## Read in Data and Basic EDA 

# In[59]:


# Austin Employment Data
pop = pd.ExcelFile('AustinEmployment.xlsx')
# Print the sheet names
print(pop.sheet_names)


# In[60]:


#read Austin Employment data into a dataframe
pop_df = pd.DataFrame(pop.parse('BLS Data Series', skiprows = 12))
pop_df.set_index('Year', inplace = True)
pop_df.head()


# In[61]:


#convert wide format into long format that is more suitable for analysis
years = []
months = []
data = []
for year in pop_df.index:
    for col in pop_df.columns:
        years.append(year)
        months.append(col)
        data.append(pop_df.loc[year][col])
df = pd.DataFrame({'Year': years, 
                   'Month': months,
                  'Employment': data})    
df.head()


# In[62]:


#use mapping to convert the Months to an integer
month_mapping = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
df['Month'] = df['Month'].map(month_mapping)


# In[122]:


#Read in Shelter Data. Going to just use the training data for this project. 
train = pd.read_csv('train.csv')
train.head()


# In[64]:


#Explore the shape of the data
train.shape


# In[65]:


#view the columns included
train.columns


# In[66]:


train[train.OutcomeType == 'Transfer']['OutcomeSubtype'].value_counts()


# In[67]:


train[train.OutcomeType == 'Euthanasia']['OutcomeSubtype'].value_counts()


# In[68]:


train[train.OutcomeType == 'Return_to_owner']['OutcomeSubtype'].value_counts()


# In[69]:


train[train.OutcomeType == 'Adoption']['OutcomeSubtype'].value_counts()


# In[70]:


train[train.OutcomeType == 'Died']['OutcomeSubtype'].value_counts()


# In[71]:


print(np.min(train.DateTime), np.max(train.DateTime))


# In[72]:


colors = train.Color.value_counts()
colors[:25]


# In[73]:


train.groupby('AnimalType').Breed.value_counts(ascending = False)


# In[74]:


print(len(train[train['AnimalType'] == 'Dog'].Breed.unique()))
train[train['AnimalType'] == 'Dog'].Breed.value_counts(ascending = False)


# In[75]:


train['SexuponOutcome'].unique()


# In[76]:


train['AgeuponOutcome'].unique()


# In[77]:


train['OutcomeType'].value_counts()


# In[78]:


train.describe()


# ## Clean Data 

# In[79]:


train.dtypes


# In[80]:


#Find Missing Values
train.isna().sum()


# In[81]:


#Remove missing values 
train = train.dropna(subset = ['AgeuponOutcome'])
train = train.dropna(subset = ['SexuponOutcome'])
train.isna().sum()


# In[82]:


#Remove unneeded columns (AnimalID and Name)
train.drop(['AnimalID', 'Name'], axis=1, inplace=True)


# In[83]:


#Change data types
train['OutcomeType'] = train['OutcomeType'].astype('category')
train['OutcomeSubtype'] = train['OutcomeSubtype'].astype('category')
train['AnimalType'] = train['AnimalType'].astype('category')
train['Breed'] = train['Breed'].astype('category')
train['DateTime'] = pd.to_datetime(train['DateTime'])


# In[84]:


#View updated data types
train.dtypes


# ## Feature Engineering

# In[85]:


#Create function to split Neutered into two fields
def NG_split(x):
        if x != 'Unknown':
            Neutered, Gender = x.split()
            if Neutered == 'Intact':
                Neutered == False
            else:
                Neutered == True
        else:
            Neutered = None
            Gender = None
        return(Neutered, Gender)


# In[86]:


Neutered = []
Gender = []
for val in train['SexuponOutcome']:
    neutered, gender = NG_split(val)
    Neutered.append(neutered)
    Gender.append(gender)
train['Neutered'] = Neutered
#convert spayed to neutered to track together
neuter_mapping = {'Neutered': 'Neutered', 'Spayed': 'Neutered', 'Intact': 'Intact', None: None}
train['Neutered'] = train['Neutered'].map(neuter_mapping)
train['Gender'] = Gender
train = train.drop('SexuponOutcome', 1)
train.head()


# In[87]:


#Create function to convert age string into a number
def agetodays(x):
        try:
            if x == 'Unknown':
                return None
            else:
                y = x.split()
        except:
            return None 
        if 'year' in y[1]:
            return float(y[0]) * 365
        elif 'month' in y[1]:
            return float(y[0]) * (365/12)
        elif 'week' in y[1]:
            return float(y[0]) * 7
        elif 'day' in y[1]:
            return float(y[0])
        
train['Age'] = train['AgeuponOutcome'].map(agetodays)
train.drop('AgeuponOutcome', axis = 'columns', inplace = True)
train.head()


# In[88]:


#Drop Missing Age values
train = train.dropna(subset=['Age', 'Neutered', 'Gender'])
train.isna().sum()


# In[89]:


colors = train.Color.value_counts()
def colorCategories(color):
    if colors[color] < 50:
        color = 'Other'
    else:
        color = color
    return(color)

train['Color'] = train['Color'].map(colorCategories)
train.head()


# In[90]:


#Create new date fields
train['Year'] = train['DateTime'].dt.year
train['Month'] = train['DateTime'].dt.month
train['Day'] = train['DateTime'].dt.day


# In[91]:


#Merge with Austin Employment data
df_train = pd.merge(train, df)
df_train.head()


# ## EDA 

# In[114]:


#Age distribution
sns.set(color_codes=True)
df_train['Age'] = df_train['Age'].astype(float)
sns.distplot(df_train['Age'], bins = 20, color = 'm')
plt.title('Distribution of Age for Shelter Animals')
plt.xlabel('Age (in days)')


# In[124]:


#What is the mean of shelter animal age?
print(df_train.Age.mean())
print(df_train.Age.mean() / 365.0)
print(df_train.Age.median())


# In[115]:


#Outcome Types Chart
df_train['OutcomeType'].value_counts(ascending = True).plot(kind = 'barh')
plt.title('Outcome Types')
plt.xlabel('Count')
plt.xlabel('Outcomes')
plt.show()


# In[121]:


#Outcome Types
df_train['OutcomeType'].value_counts(ascending = False, normalize = True)


# In[94]:


#Boxplot of Ages by Animal and gender
sns.boxplot(data=df_train,
         x='Age',
         y='AnimalType',
            hue = 'Gender')

plt.title('Animal Ages by Gender')
plt.show()


# In[95]:


#Count plot of Neutered by Male and Female animals
sns.countplot(data = df_train,
             y = "Neutered",
             hue = "Gender")
plt.show()


# In[96]:


#Count plot of Neutered by Male and Female animals
sns.countplot(data = df_train,
             y = "OutcomeType",
             hue = "Neutered")
plt.title('Outcome Type by Neutered Status')
plt.show()


# In[97]:


#Make a pivot table for Outcome Types by month (can use random value for Values since just a count)
adoptions_df = pd.pivot_table(df_train, index = 'Month', columns = 'OutcomeType', values = 'Age', aggfunc = 'count')
adoptions_df


# In[98]:


#Create a heatmap of the pivot table
sns.heatmap(adoptions_df)


# In[99]:


#Make a pivot table for Outcome Types by month and year
adoptions_df2 = pd.pivot_table(df_train, index = ['Year','Month'], columns = 'OutcomeType', values = 'Age', aggfunc = 'count').dropna()
adoptions_df2


# In[100]:


#Find row totals
#I got errors when I tried to add a column, so over-writing instead
adoptions_df3 = adoptions_df2.copy()
adoptions_df3['Adoption'] = adoptions_df2.sum(axis = 1)
adoptions_df3.drop(['Died', 'Euthanasia', 'Return_to_owner', 'Transfer'], axis = 1, inplace = True)
adoptions_df3 = adoptions_df3.rename(columns = {'Adoption': 'Total'})
adoptions_df3.reset_index(inplace = True)
adoptions_df3.head()


# In[101]:


#merge with Austin Employment data
#reset index of df so they are matching
df.reset_index(inplace = True)
#Merge two files
df_PopAdop = pd.merge(adoptions_df3, df,  how='left', left_on=['Year','Month'], right_on = ['Year','Month'])
df_PopAdop['Day'] = 1
df_PopAdop['DateTime'] = pd.to_datetime(df_PopAdop[['Year', 'Month', 'Day']])
df_PopAdop.drop(['Year', 'Month', 'Day', 'index'], axis = 1, inplace = True)
#df_PopAdop.set_index('DateTime', inplace = True)
df_PopAdop.head()


# In[102]:


#Plot results
fig = plt.figure()
ax = plt.axes()
ax.set_ylim([400, 1400])
plt.xticks(rotation=50)
plt.title('Total Adoption Events vs. Austin Employment')
plt.xlabel('Dates')
plt.ylabel('Total Adoption Events')

plt.plot(df_PopAdop.DateTime, df_PopAdop.Total)
plt.plot(df_PopAdop.DateTime, df_PopAdop.Employment)


# In[103]:


#Show the different Adoption Events over time
fig = plt.figure(figsize = (10, 6))
ax = plt.axes()
plt.xticks(rotation=50)
plt.title('Adoption Events Over Time', fontsize = 16)
plt.xlabel('Dates', fontsize = 13)
plt.ylabel('Event Counts', fontsize = 13)

plt.plot(df_PopAdop.DateTime, adoptions_df2.Adoption)
plt.plot(df_PopAdop.DateTime, adoptions_df2.Euthanasia)
plt.plot(df_PopAdop.DateTime, adoptions_df2.Died)
plt.plot(df_PopAdop.DateTime, adoptions_df2.Transfer)
plt.plot(df_PopAdop.DateTime, adoptions_df2.Return_to_owner)
plt.legend(prop = {'size': 10})


# In[104]:


# Create FacetGrid with OutcomeType
g2 = sns.FacetGrid(df_train, 
                   row="OutcomeType",
                   row_order=['Adoption', 'Transfer', 'Euthanasia', 'Return_to_owner', 'Died'],
                   size = 1, 
                   aspect = 8 )

# Map a boxplot of Age onto the grid
g2.map(sns.boxplot, 'Age')

# Show the plot
plt.show()


# ## Predictive Analytics 

# In[105]:


#Subset to only Adoption and Euthanasia as OutcomeType as these are the variables of interest. 
df_preds = df_train[(df_train['OutcomeType'] == 'Adoption') | (df_train['OutcomeType'] == 'Euthanasia')]
#Drop DateTime (repetitive) and OutcomeSubtype (confounding variable)
df_preds.drop('OutcomeSubtype', axis = 1, inplace = True)
df_preds.drop('DateTime', axis = 1, inplace = True)
df_preds.head()


# In[106]:


###CONVERT CATEGORICALS TO BINARY WITH ONE-HOT ENCODING
# limit to categorical data using df.select_dtypes()
x = df_preds.iloc[:,0:].select_dtypes(exclude=["number"])
#Convert Categorical Data to One-Hot Encoded columns
le = preprocessing.LabelEncoder()
# use df.apply() to apply le.fit_transform to all columns
x2 = x.apply(le.fit_transform)
x2.head()


# In[107]:


#combine dataframe with numeric values
df_preds = pd.concat([x2, df_preds.iloc[:,0:].select_dtypes(include=["number"])], axis=1)
df_preds.head()


# In[108]:


#Create Training and Test Split
x = df_preds.iloc[:,1:]
y = df_preds['OutcomeType']
train_data,test_data,train_label,test_label = train_test_split(x, y, test_size=0.2, random_state=42)


# ### Random Forest

# In[109]:


#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)


# In[110]:


#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(train_data,train_label)

preds=clf.predict(test_data)


# In[111]:


# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(test_label, preds))


# In[112]:


#Find Important Features
feature_imp = pd.Series(clf.feature_importances_,index=train_data.columns).sort_values(ascending=False)
feature_imp


# In[113]:


#Visualizing Important Features
sns.barplot(x=feature_imp, y=feature_imp.index)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()

