#!/usr/bin/env python
# coding: utf-8

# In[1]:


import yfinance as yfin
import pandas_datareader as pdr
import numpy_financial as npf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[4]:


#%% Question 1: Download and import stock price data

tickers = ['AMZN', 'MCD', 'WMT', 'SBUX', 'JPM']
#From 01/01/2018 until 31/12/2025.
start_date = '2018-01-01'
end_date = '2025-12-31'
# use the TICKERS function to create a ticker object for these companies
ticker = yfin.Tickers(tickers)
panel_data = ticker.history(start = start_date, end = end_date, auto_adjust=False, actions=False)
print(panel_data.columns)
# yFinance gives 'High', 'Low', 'Open', 'Close', 'Adj Close', 'Volume'.
Price = panel_data['Adj Close']
Price.head()


# In[7]:


#%% Question 2: Create a dataFrame with monthly returns for each stock.

month_end = Price.resample('ME').last()
print(month_end)
monthly_return = month_end.pct_change()
print(monthly_return)


# In[8]:


#%% Question 3: Find the volatility of each stock.

monthly_std = monthly_return.std()
print(monthly_std)


# In[13]:


#%% Question 4: Sorting Monthly_return based on volatility
sorted_volatility =  monthly_std.sort_values(ascending=False)
print(sorted_volatility)
monthly_return_sorted = monthly_return[sorted_volatility.index]
print(monthly_return_sorted)


# In[15]:


#%% Question 5: Create a portfolio
#Create a new dataframe for portfolios
portfolios = pd.DataFrame(index=monthly_return_sorted.index)

#Assigning each portfolio weights
# P1 = 100% Column 1
portfolios['P1'] = monthly_return_sorted.iloc[:,0]

# P2 = 50% Column 1 and 50% Column 2
portfolios['P2'] =  0.5 * monthly_return_sorted.iloc[:,0] + 0.5 * monthly_return_sorted.iloc[:,1]

# P3 = 33.33% Column 1, 33.33% Column 2, and 33.34% Column 3
portfolios['P3'] =  (0.3333 * monthly_return_sorted.iloc[:,0] + 
0.3333 * monthly_return_sorted.iloc[:,1] +  
0.3334 * monthly_return_sorted.iloc[:,2]
)

# P4 = 25% Column 1, 25% Column 2, 25% Column 3, and 25% Column 4
portfolios['P4'] =  (0.25 * monthly_return_sorted.iloc[:,0] + 
0.25 * monthly_return_sorted.iloc[:,1] +  
0.25 * monthly_return_sorted.iloc[:,2] +
0.25 * monthly_return_sorted.iloc[:,3]
)

# P5 = 20% Column 1, 20% Column 2, 20% Column 3, 20% Column 4, and 20% Column 5
portfolios['P5'] =  (0.20 * monthly_return_sorted.iloc[:,0] + 
0.20 * monthly_return_sorted.iloc[:,1] +  
0.20 * monthly_return_sorted.iloc[:,2] +
0.20 * monthly_return_sorted.iloc[:,3] +
0.20 * monthly_return_sorted.iloc[:,4]
)

print(portfolios)


# In[19]:


#%% Question 6: Create a bar Plot
portfolio_vol = portfolios.std()
print(portfolio_vol)

portfolio_vol.plot(kind='bar')

plt.xlabel("Number of Stocks in Portfolio")
plt.ylabel("Volatility (Standard Deviation)")
plt.title("Portfolio Volatility vs Number of Stocks")

plt.show()


# In[ ]:


#%% Question 6: Observation
# As the number of stocks in the portfolio increases, the volatility decreases. This occurs because diversification reduces unsystematic risk. 
# A portfolio with only one stock (P1) is the most volatile, but the portfolio with five stocks (P5) has the lowest volatility. 
# This demonstrates the benefit of diversification, where combining multiple assets smooths fluctuations in returns and reduces overall portfolio risk.

