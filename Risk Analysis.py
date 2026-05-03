#!/usr/bin/env python
# coding: utf-8

# In[11]:


import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[6]:


#%% Question 1: Download and import stock price data

# Define the tickers
tickers = ['AMZN', 'AAL']

# date range
start_date = '2016-01-01'
end_date = '2026-03-31'

# Download daily stock price data
stock_data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False)

# Show me the first and last 5 rows
print(stock_data.head())
print(stock_data.tail())


# In[8]:


#%% Question 2: Compute daily log returns

# Extract adjusted close prices
adj_close = stock_data['Adj Close']

# log returns
log_returns = np.log(adj_close / adj_close.shift(1))

# First row will be NaN. So, I drop the first NaN row
log_returns = log_returns.dropna()

print(log_returns.head())


# In[10]:


#%% Question 3: Variance of daily returns

# Compute variance
variance = log_returns.var()

# Assign to notation
sigma2_AMZN = variance['AMZN']
sigma2_AAL = variance['AAL']

print("σ̄²_AMZN =", sigma2_AMZN)
print("σ̄²_AAL =", sigma2_AAL)


# In[12]:


#%% Question 4: SMA volatility (45-day rolling variance)

# Rolling variance (window = 45 days)
sma_variance = log_returns.rolling(window=45).var()

# Drop initial NaNs
sma_variance = sma_variance.dropna()

# Plot
plt.figure(figsize=(10,5))
plt.plot(sma_variance.index, sma_variance['AMZN'], label='AMZN')
plt.plot(sma_variance.index, sma_variance['AAL'], label='AAL')

plt.title('45-Day SMA Volatility (Variance)')
plt.xlabel('Date')
plt.ylabel('Variance')
plt.legend()
plt.show()


# In[ ]:


# The 45-day SMA volatility shows that both AMZN and AAL experienced a significant increase in volatility during the COVID-19 period in 2020. 
# However, the increase is much more pronounced for AAL, reflecting the severe impact of the pandemic on the airline industry. 
# AAL also exhibits consistently higher volatility than AMZN throughout the sample period. 
# Additionally, volatility clustering is observed, where periods of high volatility tend to persist over time.


# In[13]:


#%% Question 5: EWMA volatility

lambda_ = 0.90

# Create columns
log_returns['EWMA_variance_AMZN'] = np.nan
log_returns['EWMA_variance_AAL'] = np.nan


# In[14]:


# Set initial values
log_returns.loc[log_returns.index[0], 'EWMA_variance_AMZN'] = sigma2_AMZN
log_returns.loc[log_returns.index[0], 'EWMA_variance_AAL'] = sigma2_AAL


# In[17]:


# For loop (for time t) that iterates from day 2 to the end of the sample

for t in range(1, len(log_returns)):
    
    # AMZN
    # Get previous AMZN variance & returns
    prev_var_amzn = log_returns.iloc[t-1]['EWMA_variance_AMZN']
    prev_ret_amzn = log_returns.iloc[t-1]['AMZN']

    log_returns.iloc[t, log_returns.columns.get_loc('EWMA_variance_AMZN')] = (
        lambda_ * prev_var_amzn + (1 - lambda_) * (prev_ret_amzn ** 2)
    )
    
    # AAL
    # Get previous AAL variance & returns
    prev_var_aal = log_returns.iloc[t-1]['EWMA_variance_AAL']
    prev_ret_aal = log_returns.iloc[t-1]['AAL']
    
    log_returns.iloc[t, log_returns.columns.get_loc('EWMA_variance_AAL')] = (
        lambda_ * prev_var_aal + (1 - lambda_) * (prev_ret_aal ** 2)
    )


# In[18]:


# Plot
plt.figure(figsize=(10,5))

plt.plot(log_returns.index, log_returns['EWMA_variance_AMZN'], label='AMZN')
plt.plot(log_returns.index, log_returns['EWMA_variance_AAL'], label='AAL')

plt.title('EWMA Volatility (λ = 0.90)')
plt.xlabel('Date')
plt.ylabel('Variance')
plt.legend()
plt.show()


# In[ ]:


#The EWMA volatility with λ = 0.90 shows that volatility reacts more quickly to market shocks compared to the SMA approach. 
#During the COVID-19 period, both stocks experienced sharp increases in volatility, with AAL exhibiting a significantly larger spike. 
#Compared to SMA, the EWMA method produces sharper peaks and faster adjustments due to its higher weighting on recent returns. 
#This demonstrates that EWMA is more responsive to sudden changes in market conditions.

