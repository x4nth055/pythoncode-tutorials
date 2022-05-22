# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from mlxtend.frequent_patterns import apriori, association_rules
from collections import Counter

# %%
# dataset = pd.read_csv("data.csv",encoding= 'unicode_escape')
dataset = pd.read_excel("Online Retail.xlsx")
dataset.head()

# %%
dataset.shape

# %%
## Verify missing value
dataset.isnull().sum().sort_values(ascending=False)

# %%
## Remove missing values
dataset1 = dataset.dropna()
dataset1.describe()

# %%
#selecting data where quantity > 0
dataset1= dataset1[dataset1.Quantity > 0]
dataset1.describe()

# %%
# Creating a new feature 'Amount' which is the product of Quantity and its Unit Price
dataset1['Amount'] = dataset1['Quantity'] * dataset1['UnitPrice']
# to highlight the Customers with most no. of orders (invoices) with groupby function
orders = dataset1.groupby(by=['CustomerID','Country'], as_index=False)['InvoiceNo'].count()
print('The TOP 5 loyal customers with most number of orders...')
orders.sort_values(by='InvoiceNo', ascending=False).head()

# %%
# Creating a subplot of size 15x6
plt.subplots(figsize=(15,6))
# Using the style bmh for better visualization
plt.style.use('bmh')
# X axis will denote the customer ID, Y axis will denote the number of orders
plt.plot(orders.CustomerID, orders.InvoiceNo)
# Labelling the X axis
plt.xlabel('Customers ID')
# Labelling the Y axis
plt.ylabel('Number of Orders')
#  Title to the plot
plt.title('Number of Orders by different Customers')
plt.show()

# %%
#Using groupby function to highlight the Customers with highest spent amount (invoices)
money = dataset1.groupby(by=['CustomerID','Country'], as_index=False)['Amount'].sum()
print('The TOP 5 profitable customers with highest money spent...')
money.sort_values(by='Amount', ascending=False).head()

# %%
# Creating a subplot of size 15*6
plt.subplots(figsize=(15,6))
# X axis will denote the customer ID, Y axis will denote the amount spent
plt.plot(money.CustomerID, money.Amount)
# Using bmh style for better visualization
plt.style.use('bmh')
# Labelling the X-axis
plt.xlabel('Customers ID')
# Labelling the Y-axis
plt.ylabel('Money spent')
# Giving a suitable title to the plot
plt.title('Money Spent by different Customers')

plt.show()

# %%
# Convert InvoiceDate from object to datetime
dataset1['InvoiceDate'] = pd.to_datetime(dataset.InvoiceDate, format='%m/%d/%Y %H:%M')
# Creating a new feature called year_month, such that December 2010 will be denoted as 201012
dataset1.insert(loc=2, column='year_month', value=dataset1['InvoiceDate'].map(lambda x: 100*x.year + x.month))
# Creating a new feature for Month
dataset1.insert(loc=3, column='month', value=dataset1.InvoiceDate.dt.month)
# Creating a new feature for Day
# +1 to make Monday=1.....until Sunday=7
dataset1.insert(loc=4, column='day', value=(dataset1.InvoiceDate.dt.dayofweek)+1)
# Creating a new feature for Hour
dataset1.insert(loc=5, column='hour', value=dataset1.InvoiceDate.dt.hour)

# %%
# Using bmh style for better visualization
plt.style.use('bmh')
# Using groupby to extract No. of Invoices year-monthwise
ax = dataset1.groupby('InvoiceNo')['year_month'].unique().value_counts().sort_index().plot(kind='bar',figsize=(15,6))
# Labelling the X axis
ax.set_xlabel('Month',fontsize=15)
# Labelling the Y-axis
ax.set_ylabel('Number of Orders',fontsize=15)
# Giving suitable title to the plot
ax.set_title('Number of orders for different Months (Dec 2010 - Dec 2011)',fontsize=15)
# Providing with X tick labels
ax.set_xticklabels(('Dec_10','Jan_11','Feb_11','Mar_11','Apr_11','May_11','Jun_11','July_11','Aug_11','Sep_11','Oct_11','Nov_11','Dec_11'), rotation='horizontal', fontsize=13)

plt.show()

# %%
# Day = 6 is Saturday.no orders placed 
dataset1[dataset1['day']==6]

# %%
# Using groupby to count no. of Invoices daywise
ax = dataset1.groupby('InvoiceNo')['day'].unique().value_counts().sort_index().plot(kind='bar',figsize=(15,6))
# Labelling X axis
ax.set_xlabel('Day',fontsize=15)
# Labelling Y axis
ax.set_ylabel('Number of Orders',fontsize=15)
# Giving suitable title to the plot
ax.set_title('Number of orders for different Days',fontsize=15)
# Providing with X tick labels
# Since there are no orders placed on Saturdays, we are excluding Sat from xticklabels
ax.set_xticklabels(('Mon','Tue','Wed','Thur','Fri','Sun'), rotation='horizontal', fontsize=15)

plt.show()

# %%
# Using groupby to count the no. of Invoices hourwise
ax = dataset1.groupby('InvoiceNo')['hour'].unique().value_counts().iloc[:-2].sort_index().plot(kind='bar',figsize=(15,6))
# Labelling X axis
ax.set_xlabel('Hour',fontsize=15)
# Labelling Y axis
ax.set_ylabel('Number of Orders',fontsize=15)
# Giving suitable title to the plot
ax.set_title('Number of orders for different Hours', fontsize=15)
# Providing with X tick lables ( all orders are placed between 6 and 20 hour )
ax.set_xticklabels(range(6,21), rotation='horizontal', fontsize=15)
plt.show()

# %%
dataset1.UnitPrice.describe()

# %%
# checking the distribution of unit price
plt.subplots(figsize=(12,6))
# Using darkgrid style for better visualization
sns.set_style('darkgrid')
# Applying boxplot visualization on Unit Price
sns.boxplot(dataset1.UnitPrice)
plt.show()

# %%
# Creating a new df of free items
freeproducts = dataset1[dataset1['UnitPrice'] == 0]
freeproducts.head()

# %%
# Counting how many free items were given out year-month wise
freeproducts.year_month.value_counts().sort_index()

# %%
# Counting how many free items were given out year-month wise
ax = freeproducts.year_month.value_counts().sort_index().plot(kind='bar',figsize=(12,6))
# Labelling X-axis
ax.set_xlabel('Month',fontsize=15)
# Labelling Y-axis
ax.set_ylabel('Frequency',fontsize=15)
# Giving suitable title to the plot
ax.set_title('Frequency for different Months (Dec 2010 - Dec 2011)',fontsize=15)
# Providing X tick labels
# Since there are 0 free items in June 2011, we are excluding it
ax.set_xticklabels(('Dec_10','Jan_11','Feb_11','Mar_11','Apr_11','May_11','July_11','Aug_11','Sep_11','Oct_11','Nov_11'), rotation='horizontal', fontsize=13)
plt.show()

# %%
plt.style.use('bmh')
# Using groupby to sum the amount spent year-month wise
ax = dataset1.groupby('year_month')['Amount'].sum().sort_index().plot(kind='bar',figsize=(15,6))
# Labelling X axis
ax.set_xlabel('Month',fontsize=15)
# Labelling Y axis
ax.set_ylabel('Amount',fontsize=15)
# Giving suitable title to the plot
ax.set_title('Revenue Generated for different Months (Dec 2010 - Dec 2011)',fontsize=15)
# Providing with X tick labels
ax.set_xticklabels(('Dec_10','Jan_11','Feb_11','Mar_11','Apr_11','May_11','Jun_11','July_11','Aug_11','Sep_11','Oct_11','Nov_11','Dec_11'), rotation='horizontal', fontsize=13)
plt.show()

# %%
# Creating a new pivot table which sums the Quantity ordered for each item
most_sold= dataset1.pivot_table(index=['StockCode','Description'], values='Quantity', aggfunc='sum').sort_values(by='Quantity', ascending=False)
most_sold.reset_index(inplace=True)
sns.set_style('white')
# Creating a bar plot of Description ( or the item ) on the Y axis and the sum of Quantity on the X axis
# We are plotting only the 10 most ordered items
sns.barplot(y='Description', x='Quantity', data=most_sold.head(10))
# Giving suitable title to the plot
plt.title('Top 10 Items based on No. of Sales', fontsize=14)
plt.ylabel('Item')

# %%
# choosing WHITE HANGING HEART T-LIGHT HOLDER as a sample
d_white = dataset1[dataset1['Description']=='WHITE HANGING HEART T-LIGHT HOLDER']

# %%
# WHITE HANGING HEART T-LIGHT HOLDER has been ordered 2028 times
d_white.shape

# %%
# WHITE HANGING HEART T-LIGHT HOLDER has been ordered by 856 customers
len(d_white.CustomerID.unique())

# %%
# Creating a pivot table that displays the sum of unique Customers who bought particular item

most_customers = dataset1.pivot_table(index=['StockCode','Description'], values='CustomerID', aggfunc=lambda x: len(x.unique())).sort_values(by='CustomerID', ascending=False)
most_customers
# Since the count for WHITE HANGING HEART T-LIGHT HOLDER matches above length 856, the pivot table looks correct for all items

# %%
most_customers.reset_index(inplace=True)
sns.set_style('white')
# Creating a bar plot of Description ( or the item ) on the Y axis and the sum of unique Customers on the X axis
# We are plotting only the 10 most bought items
sns.barplot(y='Description', x='CustomerID', data=most_customers.head(10))
# Giving suitable title to the plot
plt.title('Top 10 Items bought by Most no. of Customers', fontsize=14)
plt.ylabel('Item')

# %%
# Storing all the invoice numbers into a list y
y = dataset1['InvoiceNo']
y = y.to_list()
# Using set function to find unique invoice numbers only and storing them in invoices list
invoices = list(set(y))
# Creating empty list first_choices
firstchoices = []
# looping into list of unique invoice numbers
for i in invoices:
    
    # the first item (index = 0) of every invoice is the first purchase
    # extracting the item name for the first purchase
    firstpurchase = dataset1[dataset1['InvoiceNo']==i]['items'].reset_index(drop=True)[0]
    
    # Appending the first purchase name into first choices list
    firstchoices.append(firstpurchase)
firstchoices[:5]

# %%
# Using counter to count repeating first choices
count = Counter(firstchoices)
# Storing the counter into a datafrane
data_first_choices = pd.DataFrame.from_dict(count, orient='index').reset_index()
# Rename columns as item and count
data_first_choices.rename(columns={'index':'item', 0:'count'},inplace=True)
# Sorting the data based on count
data_first_choices.sort_values(by='count',ascending=False)

# %%
plt.subplots(figsize=(20,10))
sns.set_style('white')
# Creating a bar plot that displays Item name on the Y axis and Count on the X axis
sns.barplot(y='item', x='count', data=data_first_choices.sort_values(by='count',ascending=False).head(10))
# Giving suitable title to the plot
plt.title('Top 10 First Choices', fontsize=14)
plt.ylabel('Item')

# %%
basket = (dataset1.groupby(['InvoiceNo', 'Description'])['Quantity'].sum().unstack().reset_index().fillna(0).set_index('InvoiceNo'))
basket.head(10)

# %%
def encode_u(x):
    if x < 1:
        return 0
    if x >= 1:
        return 1

basket = basket.applymap(encode_u)
# everything is encoded into 0 and 1
basket.head(10)

# %%
# trying out on a sample item
wooden_star = basket.loc[basket['WOODEN STAR CHRISTMAS SCANDINAVIAN']==1]
# Using apriori algorithm, creating association rules for the sample item
# Applying apriori algorithm for wooden_star
frequentitemsets = apriori(wooden_star, min_support=0.15, use_colnames=True)
# Storing the association rules into rules
wooden_star_rules = association_rules(frequentitemsets, metric="lift", min_threshold=1)
# Sorting the rules on lift and support
wooden_star_rules.sort_values(['lift','support'],ascending=False).reset_index(drop=True)

# %%
# In other words, it returns the items which are likely to be bought by user because he bought the item passed into function
def frequently_bought_t(item):
    # df of item passed
    item_d = basket.loc[basket[item]==1]
    # Applying apriori algorithm on item df
    frequentitemsets = apriori(item_d, min_support=0.15, use_colnames=True)
    # Storing association rules
    rules = association_rules(frequentitemsets, metric="lift", min_threshold=1)
    # Sorting on lift and support
    rules.sort_values(['lift','support'],ascending=False).reset_index(drop=True)
    print('Items frequently bought together with {0}'.format(item))
    # Returning top 6 items with highest lift and support
    return rules['consequents'].unique()[:6]

# %%
frequently_bought_t('WOODEN STAR CHRISTMAS SCANDINAVIAN')

# %%
frequently_bought_t('JAM MAKING SET WITH JARS')

# %%



