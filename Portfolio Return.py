#!/usr/bin/env python
# coding: utf-8

# In[2]:


#%% Imported required packages for the work
import numpy as np
import pandas as pd


# In[7]:


#%% Question 1 - Access data
# Used a FOR LOOP and pandas read_excel to import Adj Close prices
# for AAPL, MSFT, TSLA, and KO from Assignment1_RawData.xlsx.

file_name = 'Assignment1_RawData.xlsx'  # Excel file name
tickers = ['AAPL', 'MSFT', 'TSLA', 'KO']

adj_close_list = []
date_series = None  

for ticker in tickers:
    # Reading each sheet by ticker name
    df_temp = pd.read_excel(file_name, sheet_name=ticker)

    # Save Date column once (all sheets have the same dates)
    if date_series is None:
        date_series = df_temp['Date']

    # Store only the Adj Close column
    adj_close_list.append(df_temp['Adj Close'])


# In[8]:


#%% Question 2 - Merge data in one NumPy array
# Convert the above DataFrames (Adj Close Series) to NumPy arrays
# using to_numpy(), then merge them with np.concatenate to obtain
# a (1258 x 4) NumPy array of prices.

adj_close_arrays = []

for series in adj_close_list:
    # Convert each Series to a 2D column vector (1258 x 1)
    adj_close_arrays.append(series.to_numpy().reshape(-1, 1))

# Merge into a single NumPy array: rows = dates, columns = stocks
prices_np = np.concatenate(adj_close_arrays, axis=1)

# For reference, prices_np has 1258 rows and 4 columns
# print(prices_np.shape)


# In[11]:


#%% Question 3 - Construct a buy-and-hold portfolio
# Equal investment of $1,000 in each stock on the first date (Jan 3, 2017).
# Allow for fractional shares. Hold the same number of shares throughout.
# Then compute the weight of each stock in the portfolio for each day.

initial_investment_per_stock = 1000.0
n_stocks = len(tickers)

# First-day prices (row 0 of prices_np)
initial_prices = prices_np[0, :]

# Number of shares of each stock to purchase (may be fractional)
nb_shares = initial_investment_per_stock / initial_prices

# Daily dollar value of each position: price * number of shares
position_values = prices_np * nb_shares  # broadcasting over rows

# Total portfolio value each day (sum across columns)
total_values = np.sum(position_values, axis=1)

# Daily portfolio weights: value of each stock / total portfolio value
weights = position_values / total_values.reshape(-1, 1)


# In[12]:


#%% Question 4 - Daily log-returns, portfolio return, and portfolio value
# 1) Compute daily log-returns for each stock:
#    r_t = ln(P_t) - ln(P_{t-1}) = diff of log prices.
# 2) Using the (time-varying) portfolio weights and these returns,
#    compute the portfolio log-return each day.
# 3) Using exponential compounding and np.cumsum, compute the
#    portfolio value over time, starting from the initial total investment.

# Log prices for each stock
log_prices = np.log(prices_np)

# Daily log-returns for each stock (rows: days, columns: stocks)
stock_log_returns = np.diff(log_prices, axis=0)

# Aligned the weights with returns (skip the first day for weights)
weights_for_returns = weights[1:, :]

# Portfolio log-return each day: sum of (weight * stock log-return)
# (similar to a sumproduct over columns)
portfolio_log_returns = np.sum(weights_for_returns * stock_log_returns, axis=1)

# Initial total portfolio value
initial_total_investment = initial_investment_per_stock * n_stocks

# Exponential compounding using cumulative sum (cumsum) of log-returns
cum_log_returns = np.cumsum(portfolio_log_returns)
portfolio_values = initial_total_investment * np.exp(cum_log_returns)


# In[14]:


#%% Question 5 - Save output to CSV
# Created a pandas DataFrame with portfolio returns and values.
# Used Date[1:] as index (because returns start from the second date).
# Then saved this DataFrame to a CSV file.

dates_for_output = date_series.to_numpy()[1:]  # drop the first date

output_df = pd.DataFrame(
    {
        'Portfolio_LogReturn': portfolio_log_returns,
        'Portfolio_Value': portfolio_values
    },
    index=dates_for_output
)

output_df.index.name = 'Date'

# Saved to CSV in the same folder as this script
output_file_name = 'Portfolio_Returns_and_Values.csv'
output_df.to_csv(output_file_name)

# Print a short confirmation and the first few rows
print('Output saved to:', output_file_name)
print(output_df.head())


# In[ ]:




