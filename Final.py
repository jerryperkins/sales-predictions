import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


df = pd.read_csv('project_data/sales_predictions.csv')

# print(df.head())
# print(df.shape)
print(df.info())

# 1. Explore the data - what do you need to do to clean this data? Are there missing values in this dataset? Some stores might not report all the data due to technical glitches or other issues. If so, deal with these appropriately.
# We need to get rid of the dupes
# We need to do something with the NAN's in "Item Weight" and "Outlet Size"
# Item_fat_content has some values that are spelled different ways ex: Low Fat and LF


#Getting rid of the dupes:

items = df['Item_Identifier']
print(items.value_counts())
df = df.drop_duplicates(subset=['Item_Identifier'])
print(df.info())

# Cleaning up Item_Fat Content
print(df['Item_Fat_Content'].value_counts())
df = df.replace({'Item_Fat_Content': {'LF' : 'Low Fat', 'reg' : 'Regular', 'low fat' : 'Low Fat'}})
print(df['Item_Fat_Content'].value_counts())


# Getting rid of Nan
print(df)
outlet_size_nans = df['Outlet_Size'].isnull()
print(df.loc[outlet_size_nans]) # There are 456 rows where a nan occurs in Outlet Size
item_weight_nans = df['Item_Weight'].isnull()
print(df.loc[item_weight_nans]) # There are 254 rows where nan occurs in Item Weight

#Option 1: Replace the Nans in the Outlet_Size column with the most common value which is Medium and remove the rows that have nan in the Item Weight Category
values = {'Outlet_Size': 'Medium'}
df_opt1 = df.fillna(value=values)
print(df_opt1)
df_opt1.dropna(inplace = True)
print(df_opt1)
temp = df_opt1.loc[:,'Item_Visibility'] < 0.0000001
df_opt1 = df_opt1.loc[~temp,:] # removing the entries with item visibility of zero


#Option 2: Remove the Outlet Size Column as it is has the most Nan's and I don't thin it is as relevant to the rest of the data. Remove the rows that have nan in the Item Weight Category.
print(df)
df_opt2 = df
df_opt2.drop(columns = ['Outlet_Size'],inplace = True)
print(df_opt2)
df_opt2.dropna(inplace=True)
print(df_opt2)

# What are some summary statistics, aggregate information, or other useful trends you can see from the data using Pandas?
print(df_opt1['Outlet_Establishment_Year'].value_counts().sort_values()) # Founding dates range over 20 years from 87-09 but only 8 of those years have a store opening. Makes me very suspicious of this column. 

print(df_opt1.groupby(['Item_Type'])[['Item_MRP']].mean().sort_values(by=['Item_MRP'])) # Starchy Foods and Seafood  have the highest MRP while Baking Goods have the lowest. Baking goods also have the largest gap b/t it and the closest item

# Data Visualization


#showing the avergae MRP of types of items as a bar graph so easily see which is the highest and lowest
mrp_mean = df_opt1.groupby(['Item_Type'])['Item_MRP'].mean().round(2).sort_values()

mrp_mean_values = list(mrp_mean.values)
mrp_mean_index = list(mrp_mean.index)

plt.rcParams['figure.figsize'] = 10,5
plt.bar(mrp_mean.index,mrp_mean.values)
plt.xticks(rotation = 'vertical')
plt.title('Average MRP of Item Types')
plt.ylabel('Dollars')
plt.xlabel('Item Types')
plt.tight_layout()
plt.show()

# displaying MRP and Outlet type sales. Was hoping to show that stores with higher MRP would result in higher sales but those numbers did not correlate all that much and that can be seen in the graphs.
outlet_sales = df_opt1.groupby(['Outlet_Identifier'])['Item_Outlet_Sales'].mean().sort_values()
outlet_item_mrp = df_opt1.groupby(['Outlet_Identifier'])['Item_MRP'].mean()
combo = pd.concat([outlet_sales, outlet_item_mrp], axis = 1) #combined data frames so that I could plot mrp and outlet sales and have them both be associated with the correct outlet type
outlet_sales = combo['Item_Outlet_Sales']
outlet_item_mrp = combo['Item_MRP']
# print(outlet_sales, outlet_item_mrp)
outlet_ids = list(outlet_sales.index)
outlet_sales = list(outlet_sales.values)
outlet_item_mrp = list(outlet_item_mrp.values)
# print(outlet_item_mrp)
test = np.arange(8)
w = .3

plt.rcParams['figure.figsize'] = 10,5
plt.bar(test, outlet_sales, align = 'center', width = w, label = 'Outlet Sales', color = 'b')
plt.xticks(np.arange(8), outlet_ids)
plt.ylabel('Outlet Sales in Dollars')
plt.xlabel('Outlet Types')
plt.legend(loc=(1.04, 0.06))
plt.tight_layout()
#second Y axis
plt.twinx()
plt.bar(test + w, outlet_item_mrp, align = 'center', width = w, color = 'r', label = 'MRP')
plt.title('Average Outlet Sales and MRP')
plt.ylabel('MRP in Dollars')
plt.legend(loc=(1.04,0))
plt.tight_layout()

plt.show()


# Here is where Part4 begins

df_opt1['Item_Visibility'].hist() # Over half of the items have less than 10% visibility
plt.xlabel('Visibility')
plt.show()

visibility_avg = df_opt1.groupby(['Item_Type'])['Item_Visibility'].mean().sort_values() # the second lowest MRP item group has the highest visibility percentage by almost 30%...ouch
plt.bar(visibility_avg.index, visibility_avg.values)
plt.xticks(rotation=45, ha='right')
plt.title('Avg Visibility Per Item Type')
plt.ylabel('Visibility')
plt.tight_layout()
plt.show()

#Boxplots for item mrp dist of each item type
item_types = df_opt1['Item_Type'].value_counts().index
index_list = []
plt.rcParams['figure.figsize'] = 10,5
for index, value in enumerate(item_types):
    item_type = df_opt1[df_opt1['Item_Type']==value]
    # print(len(item_type))
    # print(len(item_type['Item_MRP']))
    # print(np.arange(len(item_type['Item_MRP'])))
    plt.boxplot(item_type['Item_MRP'],
                positions=[index],
                widths=.6,
                medianprops=dict(linestyle='-', linewidth=2, color='green'),
                notch=True,
                showmeans=True,
                meanprops =dict(marker='X', markeredgecolor='black', markerfacecolor='r'))
    # item_type['Item_MRP'].hist(color='r', alpha=.5)
    index_list.append(value)
plt.xticks(range(0,len(item_types)) , item_types, rotation=45, ha='right') #the idea for using range came from here: https://stackoverflow.com/questions/58814857/conversionerror-failed-to-convert-values-to-axis-units
plt.title("Item MRP distribution Per Item Type")
plt.ylabel('Item MRP')
plt.tight_layout()
plt.show()

# Boxplots for item visibility distribution for each item type
item_types = df_opt1['Item_Type'].value_counts().index
index_list = []
plt.rcParams['figure.figsize'] = 10,5
for index, value in enumerate(item_types):
    item_type = df_opt1[df_opt1['Item_Type']==value]
    # print(len(item_type))
    # print(len(item_type['Item_MRP']))
    # print(np.arange(len(item_type['Item_MRP'])))
    plt.boxplot(item_type['Item_Visibility'],
                positions=[index],
                widths=.6,
                medianprops=dict(linestyle='-', linewidth=2, color='green'),
                notch=True,
                showmeans=True,
                meanprops =dict(marker='X', markeredgecolor='black', markerfacecolor='r'))
    # item_type['Item_MRP'].hist(color='r', alpha=.5)
plt.xticks(range(0,len(item_types)) , item_types, rotation=45, ha='right') #the idea for using range came from here: https://stackoverflow.com/questions/58814857/conversionerror-failed-to-convert-values-to-axis-units
plt.title("Item Visibility distribution Per Item Type")
plt.ylabel('Item Visibility')
plt.tight_layout()
plt.show()


#PART 5 STARTS HERE
#Regression without fat content or item type


X = df_opt1.loc[:, ['Item_Weight', 'Item_Visibility', 'Item_MRP']].values
print(X, X.shape)
y = df_opt1.loc[:, 'Item_Outlet_Sales'].values
print(y, y.shape)

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
print(X)

reg = LinearRegression(fit_intercept=True)
reg.fit(X,y)
score = reg.score(X,y)
print(score) # score = 0.3754456883280959 which is not great

m1 = reg.coef_[0]
m2 = reg.coef_[1]
m3 = reg.coef_[2]
# print(m1, m2, m3)
b = reg.intercept_

# print("formula: y = {:.2f}*Item_Weight + {:.2f}*Item_Visibility + {:.2f}*Item_MRP + {:.2f}".format(m1,m2,m3,b) ) # item visibility seems wonky so I am going to look more into that.

# print(df_opt1['Item_Visibility'].value_counts()) # turns out 74 values are equal to zero. I am guessing this is bad data that I missed previously as every item at a store has to be at least a little visbile to be purchased.

# Going to do regression using the same columns again but this time with the rows that have item visibility = 0 removed

temp = df_opt1.loc[:,'Item_Visibility'] < 0.0000001
# print(temp)
df_reg = df_opt1.loc[:,['Item_Weight','Item_Visibility', 'Item_MRP', 'Item_Outlet_Sales']]
df_reg = df_reg.loc[~temp,:]
# print(df_reg)

XX = df_reg.loc[:, ['Item_Weight','Item_Visibility', 'Item_MRP']].values
# print(XX, XX.shape)
yy = df_reg.loc[:, 'Item_Outlet_Sales'].values
# print(yy, yy.shape)

scaler = StandardScaler()
scaler.fit(XX)
XX = scaler.transform(XX)
# print(XX)

reg = LinearRegression(fit_intercept=True)
reg.fit(XX,yy)
score_XX = reg.score(XX,yy)
print(score_XX) #0.3817810129168757 is the score with the zero visibility scores removed. It improved a little but not much

m1 = reg.coef_[0]
m2 = reg.coef_[1]
m3 = reg.coef_[2]
# print(m1, m2, m3)
b = reg.intercept_

# print("formula: y = {:.2f}*Item_Weight + {:.2f}*Item_Visibility + {:.2f}*Item_MRP + {:.2f}".format(m1,m2,m3,b) ) # Item visibility actually had an even higher impact on lowering item sales


# Going to do Regression including Categorical Variables of fat content and item type
df_opt1 = df_opt1.loc[~temp,:] # removing the entries with item visibility of zero
df_cat_reg = df_opt1.loc[:,['Item_Weight','Item_Visibility', 'Item_MRP', 'Item_Fat_Content', 'Item_Type']]

df_cat_reg = pd.get_dummies(df_cat_reg, columns=['Item_Fat_Content', 'Item_Type'])
X = df_cat_reg.loc[:,:].values
# print(X,X.shape)
y = df_opt1.loc[:, 'Item_Outlet_Sales'].values
# print(y, y.shape)

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
# print(X)

reg = LinearRegression(fit_intercept=True)
reg.fit(X,y)
score = reg.score(X,y)
print(score) # 0.3893874828616616 is higher than not using the categorical variables but still not a great score

columns = df_cat_reg.columns
formula = "formula: y ="
for i in range (0, len(X[0]), 1):
    # print(columns[i], reg.coef_[i].round(2)) # list of coefficients so we can see how each thing is effecting sales prediction
    formula = formula + " " + str(reg.coef_[i].round(2)) + "*" + columns[i] + " + "
formula = formula + str(reg.intercept_.round(2)) # formula because I did not want to write the whole thing out
print(formula)


# Regression adding in categorical variables of outlet location and outlet type

df_cat_reg = df_opt1.loc[:,['Item_Weight','Item_Visibility', 'Item_MRP', 'Item_Fat_Content', 'Item_Type', 'Outlet_Location_Type', 'Outlet_Type' ]]

df_cat_reg = pd.get_dummies(df_cat_reg, columns=['Item_Fat_Content', 'Item_Type', 'Outlet_Location_Type', 'Outlet_Type'], drop_first=True)
X = df_cat_reg.loc[:,:].values
# print(X,X.shape)
y = df_opt1.loc[:, 'Item_Outlet_Sales'].values
# print(y, y.shape)

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
# print(X)

reg = LinearRegression(fit_intercept=True)
reg.fit(X,y)
score = reg.score(X,y)
print(score) # 0.4977001608966307 is 25% higher than the score we got without Outlet_Location_Type and Outlet_Type
predictions = reg.predict(X)
print(mean_absolute_error(y, predictions)) #808.0576459575786
print(mean_squared_error(y,predictions)) #1189027.823266993
print(np.sqrt(mean_squared_error(y,predictions))) #1090.4255239432875

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=3)

# linear reg with train_test_split
reg.fit(X_train, y_train)
predictions = reg.predict(X_test)
score = reg.score(X_test, y_test)
print(score) #0.39007641479345
print(mean_absolute_error(y_test, predictions)) # 901.9496424405412
print(mean_squared_error(y_test,predictions)) # 1456323.7721069388
print(np.sqrt(mean_squared_error(y_test,predictions))) # 1206.782404622697

columns = df_cat_reg.columns
formula = "formula: y ="
data = {}
for i in range (0, len(X[0]), 1):
    data[columns[i]] = reg.coef_[i].round(2) # tossing our coefficients into a dictionary
    # print(columns[i], reg.coef_[i].round(2)) # list of coefficients so we can see how each thing is effecting sales prediction. Item visibility went from a coefficient of -1915.29 to 334.28 when we added Outlet_Location_Type and Outlet_Type
    formula = formula + " " + str(reg.coef_[i].round(2)) + "*" + columns[i] + " + "
formula = formula + str(reg.intercept_.round(2)) # formula because I did not want to write the whole thing out ;)
# features_df = pd.DataFrame({'feature': X_train.columns, 'importance': rf_reg.feature_importances_})
print(formula)
print(data.keys())
reg_importance_df = pd.DataFrame({'feature': data.keys(), 'importance': data.values()})
reg_importance_df.sort_values(by='importance', inplace=True)
print(reg_importance_df)

plt.figure(figsize=(10,5))
plt.barh(reg_importance_df['feature'], reg_importance_df['importance'])
plt.tight_layout()
plt.show()

data_sorted = sorted(data.items(), key=lambda x: x[1], reverse=True)# https://stackoverflow.com/questions/613183/how-do-i-sort-a-dictionary-by-value
# print(data_sorted) # top 5 influencers on sales are: Outlet_Type_Supermarket Type1, Outlet_Type_Supermarket Type2, Item_Type_Seafood, Item_Visibility, Item_Type_Others


# # # regression using KNN

df_knn_reg = df_opt1.loc[:,['Item_Weight','Item_Visibility', 'Item_MRP', 'Item_Fat_Content', 'Item_Type', 'Outlet_Location_Type', 'Outlet_Type' ]]

df_knn_reg = pd.get_dummies(df_knn_reg, columns=['Item_Fat_Content', 'Item_Type', 'Outlet_Location_Type', 'Outlet_Type'])
X = df_knn_reg.loc[:,:].values
# print(X,X.shape)
y = df_opt1.loc[:, 'Item_Outlet_Sales'].values
# print(y, y.shape)

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
# print(X)

knn_reg = KNeighborsRegressor(n_neighbors=5)
knn_reg.fit(X,y)
score = knn_reg.score(X,y)
print(score) #0.5034447000673261 is the highest score of all the different models created



## baggedrgressor 
print(X.shape)
X = df_knn_reg
y = df_opt1['Item_Outlet_Sales'].values
print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=3)

bag_reg = BaggingRegressor(n_estimators=10)
bag_reg.fit(X_train, y_train)
score = bag_reg.score(X_test, y_test)
print(score) # 0.3650347782444854



#random forest regressor

rf_reg = RandomForestRegressor(n_estimators=100, bootstrap=True, oob_score=True)
rf_reg.fit(X_train, y_train)
score = rf_reg.score(X_test, y_test)
print(score) #0.38338648968218136


features_df = pd.DataFrame({'feature': X_train.columns, 'importance': rf_reg.feature_importances_})
features_df.sort_values(by='importance', inplace=True)
print(features_df)

plt.figure(figsize=(10,5))
plt.barh(features_df['feature'], features_df['importance'])
plt.title("Feature Importance")
plt.xlabel('Importance')
plt.tight_layout()
plt.show()


OI = df_opt1.loc[:, 'Outlet_Identifier'] == 'OUT010'
OI = df_opt1.loc[OI, :]
print(OI['Outlet_Type'].value_counts()) #shows that all of the grocery stores are out10

print(df_opt1['Outlet_Type'].value_counts())#shows that all of the grocery stores are out10

