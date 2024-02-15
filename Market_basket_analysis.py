import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
import networkx as nx
from mlxtend.frequent_patterns import apriori, association_rules
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly


# Load the dataset
df = DataFrame()
path = "C:/Users/aband/OneDrive/Desktop/BTU_AI/Data_Exploration/Bakery sales.csv"
df = pd.read_csv(path)

# Explore and preprocess the data
print(df.head())
print("\nDataframe info: index dtype and columns, non-null values and memory usage\n")
print(df.info())
print("\n data type of each column\n")
print(df.dtypes)
print("\nDescriptive statistics:\n")
print(df.describe())
print(df.describe(include=object))

# Convert the 'unit_price' column to numeric (remove the '€' symbol)
print("\nConverting the 'unit_price' column to numeric (remove the '€' symbol)...\n")
df['unit_price'] = pd.to_numeric(df['unit_price'].str.replace(',', '.').str.replace(' €', ''))

# Check for missing values and handling them if necessary
print("\nChecking for missing values and handling them if necessary...\n")
print(df.isnull().sum())

# Remove duplicates if any
print("\nRemoving duplicates if any...\n")
df = df.drop_duplicates()
print(df.head())


itemFrequency = df['article'].value_counts().sort_values(ascending=False)
itemFrequency.head(20)

fig = px.bar(itemFrequency.head(20), title='20 Most Frequent Items', color=itemFrequency.head(20), color_continuous_scale=px.colors.sequential.Mint)
fig.update_layout(margin=dict(t=50, b=0, l=0, r=0), titlefont=dict(size=20), xaxis_tickangle=-45, plot_bgcolor='white', coloraxis_showscale=False)
fig.update_yaxes(showticklabels=False, title=' ')
fig.update_xaxes(title=' ')
fig.update_traces(texttemplate='%{y}', textposition='outside', hovertemplate = '<b>%{x}</b><br>No. of Transactions: %{y}')
fig.show()

# Calculate the revenue for each product
df['revenue'] = df['Quantity'] * df['unit_price']
print(f"\n{df.head(10)}\n")

# Group by article and sum the revenue
product_revenue = df.groupby('article')['revenue'].sum()
print(f"\n{product_revenue}\n")

# Select the top 20 products with the most revenue
top_20_revenue_products = product_revenue.nlargest(20).index

# Filter the dataframe to include only the top 20 revenue products
df_top_20_revenue = df[df['article'].isin(top_20_revenue_products)]

# Plot the top 20 revenue products with total revenue on top of each bar
fig = px.bar(df_top_20_revenue, x='article', y='revenue', title='Top 20 Products with the Most Revenue',
             color='revenue', color_continuous_scale=px.colors.sequential.Mint,
             text='revenue', labels={'revenue': 'Total Revenue'})
fig.update_layout(margin=dict(t=50, b=0, l=0, r=0), titlefont=dict(size=20), xaxis_tickangle=-45,
                  plot_bgcolor='white', coloraxis_showscale=False)
fig.update_yaxes(showticklabels=False, title=' ')
fig.update_xaxes(title=' ')
fig.update_traces(texttemplate='%{text:.2f}', textposition='outside',
                  hovertemplate='<b>%{x}</b><br>Revenue: $%{y:.2f}')
fig.show()

# Extract the hour from the 'time' column
df['hour'] = pd.to_datetime(df['time']).dt.hour

# Group by hour and sum the quantities
hourly_sales = df.groupby('hour')['Quantity'].sum().reset_index()

# Plot the peak hours of sale
fig = px.bar(hourly_sales, x='hour', y='Quantity', title='Peak Hours of Sale',
             labels={'hour': 'Hour of the Day', 'Quantity': 'Total Quantity Sold'},
             color='Quantity', color_continuous_scale=px.colors.sequential.Mint,
             text='Quantity')
fig.update_layout(margin=dict(t=50, b=0, l=0, r=0), titlefont=dict(size=20),
                  xaxis_tickangle=-45, plot_bgcolor='white', coloraxis_showscale=False)
fig.update_yaxes(showticklabels=False, title=' ')
fig.update_xaxes(title=' ')
fig.update_traces(texttemplate='%{text}', textposition='outside',
                  hovertemplate='<b>Hour</b>: %{x}<br>Total Quantity Sold: %{y}')
fig.show()

# Group by article and sum the quantities
product_quantities = df.groupby('article')['Quantity'].sum()

# Select the 5 least sold products
least_sold_products = product_quantities.nsmallest(5).index

# Filter the dataframe to include only the 5 least sold products
df_least_sold = df[df['article'].isin(least_sold_products)]

# Plot the 5 least sold products
fig = px.bar(df_least_sold, x='article', y='Quantity', title='5 Least Sold Products',
             labels={'article': 'Product', 'Quantity': 'Total Quantity Sold'},
             color='Quantity', color_continuous_scale=px.colors.sequential.Mint,
             text='Quantity')
fig.update_layout(margin=dict(t=50, b=0, l=0, r=0), titlefont=dict(size=20),
                  xaxis_tickangle=-45, plot_bgcolor='white', coloraxis_showscale=False)
fig.update_yaxes(showticklabels=False, title=' ')
fig.update_xaxes(title=' ')
fig.update_traces(texttemplate='%{text}', textposition='outside',
                  hovertemplate='<b>Product</b>: %{x}<br>Total Quantity Sold: %{y}')
fig.show()

#Create “single_transaction” column (combination of the ticket number and date)
#Tells us the item purchased in one receipt.
print("\nCreating “single_transaction” column (combination of the ticket number and date). It Tells us the item purchased in one receipt...\n")
df['single_transaction'] = df['ticket_number'].astype(str)+'_'+df['date'].astype(str)

#pivot this table to convert the items into columns and the transaction into rows
print("\nconverting the items into columns and the transaction into rows...\n")
df2 = pd.crosstab(df['single_transaction'], df['article'])

#encoding all values in the data frame to 0 and 1.
def encode(item_freq):
    res = 0
    if item_freq > 0:
        res = 1
    return res
    
basket_input = df2.applymap(encode)

# Generate frequent itemsets using Apriori algorithm
print("\nGenerating frequent itemsets using Apriori algorithm...\n")
frequent_itemsets = apriori(basket_input, min_support=0.001, use_colnames=True)

# Generate association rules
print("\nGenerating association rules...\n")
rules = association_rules(frequent_itemsets, metric="lift")
print(f"\n {rules.head(20)} \n")

network_A = list(rules["antecedents"].unique())
network_B = list(rules["consequents"].unique())
node_list = list(set(network_A + network_B))
G = nx.Graph()
for i in node_list:
    G.add_node(i)
for i,j in rules.iterrows():
    G.add_edges_from([(j["antecedents"], j["consequents"])])
pos = nx.spring_layout(G, k=0.5, dim=2, iterations=400)
for n, p in pos.items():
    G.nodes[n]['pos'] = p

edge_trace = go.Scatter(x=[], y=[], line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')

for edge in G.edges():
    x0, y0 = G.nodes[edge[0]]['pos']
    x1, y1 = G.nodes[edge[1]]['pos']
    edge_trace['x'] += tuple([x0, x1, None])
    edge_trace['y'] += tuple([y0, y1, None])

node_trace = go.Scatter(x=[], y=[], text=[], mode='markers', hoverinfo='text',
    marker=dict(showscale=True, colorscale='Burg', reversescale=True, color=[], size=15,
    colorbar=dict(thickness=10, title='Node Connections', xanchor='left', titleside='right')))

for node in G.nodes():
    x, y = G.nodes[node]['pos']
    node_trace['x'] += tuple([x])
    node_trace['y'] += tuple([y])

for node, adjacencies in enumerate(G.adjacency()):
    node_trace['marker']['color']+=tuple([len(adjacencies[1])])
    node_info = str(adjacencies[0]) +'<br>No of Connections: {}'.format(str(len(adjacencies[1])))
    node_trace['text']+=tuple([node_info])

fig = go.Figure(data=[edge_trace, node_trace], 
    layout=go.Layout(title='Item Connections Network', titlefont=dict(size=20),
    plot_bgcolor='white', showlegend=False, margin=dict(b=0,l=0,r=0,t=50),
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))

iplot(fig)

#sort the dataset by support, confidence, and lift
print("\nsorting the dataset by support, confidence, and lift...\n")
print(rules.sort_values(["support", "confidence","lift"],axis = 0, ascending = False).head(20))

# Provide recommendations for production and storage
# (Consider rules with high lift and confidence, and also high support)
production_recommend = rules[(rules['lift'] > 1) & (rules['confidence'] > 0.5)]
production_recommendations = rules[(rules['antecedent support'] > 0.05) & (rules['consequent support'] > 0.05)]
print("\nProduction Recommendations based on highest support:\n")
print(production_recommendations[['antecedents', 'consequents', 'antecedent support', 'consequent support']])


### Machine Learning ####

# Combine 'date' and 'time' columns into a single datetime column
df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])

# Extract the relevant columns
time_series_data = df.groupby('datetime')['Quantity'].sum().reset_index()

# Set 'datetime' as the index
time_series_data.set_index('datetime', inplace=True)

# Plot the original time series data
plt.figure(figsize=(12, 6))
plt.plot(time_series_data.index, time_series_data['Quantity'], label='Actual Quantity')
plt.title('Time Series Plot of Quantity Sold Over Time')
plt.xlabel('Datetime')
plt.ylabel('Quantity')
plt.legend()
plt.show()


# Set the DateTime as the index
df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
df.set_index("datetime", inplace=True)  # Set datetime as the index
df.index = pd.to_datetime(df.index)

# Feature Engineering
df['year'] = df.index.year
df['month'] = df.index.month
df['day'] = df.index.day
df['hour'] = df.index.hour

# Prepare data for Prophet
prophet_data = df.groupby(['datetime', 'article']).sum()['Quantity'].reset_index()
prophet_data.rename(columns={'datetime': 'ds', 'Quantity': 'y'}, inplace=True)

# Determine the cut-off date for training and testing split
cut_off_date = prophet_data['ds'].iloc[-365 * 24]  # Assuming you want to keep the last year as a test set

# Split the data into training and testing sets
train = prophet_data[prophet_data['ds'] <= cut_off_date]
test = prophet_data[prophet_data['ds'] > cut_off_date]

# Model Training
model = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=True)
model.add_seasonality(name='hourly', period=24, fourier_order=3)
model.fit(train)

# Make future dataframe for prediction
future = model.make_future_dataframe(periods=len(test), freq='H')

# Predictions
forecast = model.predict(future)

# Plot the forecast
fig = model.plot(forecast, xlabel='Datetime', ylabel='Quantity', figsize=(14, 8))
fig2 = model.plot_components(forecast)
fig.show()
fig2.show()

# Plot the original data with the cut-off line
plt.figure(figsize=(15, 5))
plt.plot(df.index, df['Quantity'], linestyle="-", label="Original Data")
plt.axvline(x=cut_off_date, color='r', linestyle='--', label="Cut-off Date")
plt.title("Original Data with Cut-off Date")
plt.ylabel('Quantity')
plt.xlabel('Datetime')
plt.legend()
plt.show()
