import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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

visibility_avg = df_opt1.groupby(['Item_Type'])['Item_Visibility'].mean().sort_values() # the second lowest MRP item group has the highest visibility percentage by almost 30%...ouch
plt.bar(visibility_avg.index, visibility_avg.values)
plt.xticks(rotation=45, ha='right')
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


