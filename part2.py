import pandas as pd

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



