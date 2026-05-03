#!/usr/bin/env python
# coding: utf-8

# In[40]:


# Importing the required libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random


# In[41]:


#PART A
#Question 1: Import XIU_prices.csv and XIU_weight_matrix.csv into Dataframes

df_xiu_prices = pd.read_csv('XIU_prices.csv')
df_xiu_weights = pd.read_csv('XIU_weight_matrix.csv')


#Set Date as index
df_xiu_prices.set_index('Date', inplace=True)
print(df_xiu_prices)

df_xiu_weights.set_index('Date', inplace=True)
print(df_xiu_weights)


# In[42]:


#Question 2: Compute monthly percentage return from xiu_prices.
df_xiu_returns = df_xiu_prices.pct_change()
df_xiu_returns


# In[43]:


#Question 3: Displaying the first 12 months  from df_xiu_weights
df_xiu_weights.head(12)


# In[44]:


#Question 3: Displaying the last 12 months  from df_xiu_weights
df_xiu_weights.tail(12)


# In[45]:


#Converting Date to datetime for efficient indexing
df_xiu_prices.index = pd.to_datetime(df_xiu_prices.index)
df_xiu_weights.index = pd.to_datetime(df_xiu_weights.index)


# In[46]:


#Question 3: Identify
#Firms originally included in 2015-2016
firms_2015_2016 = set(df_xiu_weights.loc['2015':'2016'].columns[(df_xiu_weights.loc['2015':'2016'] !=0).any()])
firms_2015_2016

#Firms in 2017 and beyond
firms_2017_onward = set(df_xiu_weights.loc['2017':].columns[(df_xiu_weights.loc['2017':] !=0).any()])
firms_2017_onward

#Firms originally included in 2015-2016 but later removed

removed_firms = firms_2015_2016 - firms_2017_onward
print('Removed Firms:', removed_firms)

#Firms newly added in later years
new_firms = firms_2017_onward - firms_2015_2016
print('New Firms:', new_firms)


# In[47]:


#PART B
#Question 1: Extract the weights as of 2025-10-31 into a new DataFrame, df_weights. 
df_weights = df_xiu_weights.loc['2025-10-31']
print(df_weights)


# In[49]:


#Question 2
#Ensuring ticker alignment
common_tickers = df_xiu_returns.columns.intersection(df_weights.index)
df_returns = df_xiu_returns[common_tickers]
df_weights = df_weights[common_tickers]


# In[50]:


#Question 3
#Renormalize the weights
df_weights = df_weights / df_weights.sum()
print('Sum of weights after renormalization:', df_weights.sum())
print(df_weights)


# In[51]:


#Question 4
#Calculate the portfolio returns and a new column added named "Portfolio"
df_returns['Portfolio'] = (df_returns * df_weights).sum(axis=1)
print(df_returns)


# In[52]:


#Question 5
#Scatter Plot
plt.scatter(df_xiu_returns['XIU.TO'], df_returns['Portfolio'])

# Labels and title
plt.xlabel('Actual XIU.TO Returns')
plt.ylabel('Reconstructed Portfolio Returns')
plt.title('Scatter Plot of Reconstructed vs Actual XIU Returns')
plt.show()


# In[ ]:


#Comment
The scatter plot follows 45-degree line, which shows that there is a very strong positive relationship between the reconstructed portfolio returns 
and the actual XIU.TO returns. 

This expected to happen in theory since the portfolio is constructed by the same weights of the constituents and 
return data that underlie the index. With perfect reconstruction, the exact index methodology, using accurate weights and complete tickers, then 
each monthly return must perfectly match XIU.TO.

However, deviations can occur due to real-world factors such as uncomplete coverage of tickers, static weights such as the reconstruction are based on 
weights of a single date, whereas XIU is periodically rebalanced. Such a mismatch presents 
tracking error. Some incidences such as stock splits, dividends or mergers can influence returns differently between the index and the raw price.

The differences in timing and rounding can also be significant that a difference of even a few cents in the calculation of returns or the rounding 
of weights can cause a noticeable variance.


# In[53]:


#Question 6: Correlation between reconstructed portfolio and Actual (XIU.TO)
correlation = df_returns['Portfolio'].corr(df_xiu_returns['XIU.TO'])
print('Correlation betweeen the is:', round(correlation,4))


# In[ ]:


#Comment
The correlation between the reconstructed portfolio and XIU.TO is 0.9421, indicating a strong positive relationship. 
This further confirmed that the reconstructed portfolio returns closely tracks the actual index performance.


# In[54]:


#PART C
#Question 1
#Renormalize weights (same as used in Part B)
df_weights = df_weights / df_weights.sum()
print('Sum of weights after renormalization:', df_weights.sum())
print(df_weights)


# In[55]:


#Question 2: Calculate the mean and covariance matrix of the 58 stocks.
mean_vec = df_returns.iloc[:,:-1].mean()
cov_matrix = df_returns.iloc[:,:-1].cov()
print("Mean Vector")
print(mean_vec)
print("\nCovariance Matrix")
print(cov_matrix)


# In[56]:


#Question 3:
vec = np.arange(1,59)
er_vec = []
std_vec = []

for k in vec:

    # Select first k stocks by column position
    returns_sub = df_returns.iloc[:, :k]

    # XIU weights corresponding to the same first k stocks
    x = df_weights.iloc[:k]
    x = x / x.sum()

    # Covariance matrix
    covariance_matrix = returns_sub.cov()

    # Expected return using first k mean returns
    er = x @ mean_vec.iloc[:k]

    # Standard deviation
    std = np.sqrt(x @ covariance_matrix @ x.T)

    er_vec.append(er)
    std_vec.append(std)

print("Expected Returns:")
print(er_vec)

print("\nStandard Deviations:")
print(std_vec)


# In[57]:


#Question 4:
vec = np.arange(1,59)
ER_vec = []   
STD_vec = []

for k in vec:

    returns_sub = mean_vec.index[:k]

    w = df_weights[returns_sub]
    w = w / w.sum()
    
    er = np.dot(w, mean_vec[returns_sub])

    cov_sub = cov_matrix.loc[returns_sub, returns_sub]
    std = np.sqrt(np.dot(w.T, np.dot(cov_sub, w)))
    
    ER_vec.append(er)
    STD_vec.append(std)

print("Expected Returns:")
print(ER_vec)

print("\nStandard Deviations:")
print(STD_vec)


# In[58]:


#Question 5: Plot portfolio standard deviation against the number of stocks
# The Number of stocks (1 to 58)
num_stocks = vec
plt.plot(num_stocks, STD_vec)

plt.xlabel('Number of Stocks in Portfolio')
plt.ylabel('Portfolio Standard Deviation')
plt.title('Diversification Effect: Portfolio Risk vs Number of Stocks')
plt.show()


# In[ ]:


#Question 6
#Comment
This pattern in the plot matches the principle of diversification discussed in the lecture. 
The standard deviation of the portfolio decreases rapidly as the number of stocks in the portfolio increases beyond 1 to the range of 10. 
It is a signal of the high rate at which the idiosyncratic (firm-specific) risk is reduced with the incorporation of additional, uncorrelated stocks. 
After 10 stocks, the decline in risk increasingly slow and ultimately stabilizes, and this implies that the greater part of the diversification 
advantage has already been achieved.


# In[59]:


#PART D
#Question 1: Generate 5,000 Random Portfolio using First 3 Stocks
# Generate random weights
def weights(n, x_min, x_max):
    w = np.random.uniform(x_min, x_max, n)   # random numbers in [x_min, x_max]
    w = w / w.sum()                          # renormalize to sum 1
    return w

# Portfolio setup
N = 3                  # first 3 stocks
x_min = 0
x_max = 1
N_portf = 5000

# Select first 3 stocks
selected_stocks = df_returns.columns[:N]

# Extract mean returns and covariance for these stocks
mu = mean_vec[selected_stocks]                       
cov = cov_matrix.loc[selected_stocks, selected_stocks]  

# Lists to store results
ER_vec = []
STD_vec = []
weight_vec = []

# Loop to generate portfolios
for i in range(N_portf):
    w = weights(N, x_min, x_max)     # random weights
    weight_vec.append(w)             # store weights

    # Compute expected return
    er = w @ mu

    # Compute standard deviation
    std = np.sqrt(w @ cov @ w.T)

    # Store results
    ER_vec.append(er)
    STD_vec.append(std)


# In[60]:


#Question 2
rf = 0.0025

# Save results into a DataFrame
df_portfolio_3 = pd.DataFrame({
    'Expected_Return': ER_vec,
    'Std_Dev': STD_vec,
    'Weights': weight_vec
})

# Compute Sharpe ratio
df_portfolio_3['Sharpe_Ratio'] = (df_portfolio_3['Expected_Return'] - rf) / df_portfolio_3['Std_Dev']

df_portfolio_3


# In[61]:


#Question 3
# Generate random weights
def weights(n, x_min, x_max):
    w = np.random.uniform(x_min, x_max, n)
    w = w / w.sum()  # renormalize to sum 1
    return w

# Portfolio setup
N = 6                  # first 6 stocks
x_min = 0
x_max = 1
N_portf = 10000
rf = 0.0025            # risk-free rate

# Select first 6 stocks
selected_stocks = df_returns.columns[:N]

# Extract mean returns and covariance
mu = mean_vec[selected_stocks]
cov = cov_matrix.loc[selected_stocks, selected_stocks]

# Lists to store results
ER_vec_6 = []
STD_vec_6 = []
weight_vec_6 = []

# Loop to generate portfolios
for i in range(N_portf):
    w = weights(N, x_min, x_max)       # random weights
    weight_vec_6.append(w)

    er = w @ mu
    std = np.sqrt(w @ cov @ w.T)

    ER_vec_6.append(er)
    STD_vec_6.append(std)

# Save results into a DataFrame
df_portfolio_6 = pd.DataFrame({
    'Expected_Return': ER_vec_6,
    'Std_Dev': STD_vec_6,
    'Weights': weight_vec_6
})

# Compute Sharpe ratio
df_portfolio_6['Sharpe_Ratio'] = (df_portfolio_6['Expected_Return'] - rf) / df_portfolio_6['Std_Dev']

df_portfolio_6


# In[62]:


#Question 4

# Plot 3-stock portfolios (blue circles)
plt.scatter(df_portfolio_3['Std_Dev'], df_portfolio_3['Expected_Return'],
            c='blue', alpha=0.5, label='3-stock portfolios')

# Plot 6-stock portfolios (red triangles)
plt.scatter(df_portfolio_6['Std_Dev'], df_portfolio_6['Expected_Return'],
            c='red', alpha=0.5, label='6-stock portfolios')

# Labels and title
plt.xlabel('Portfolio Standard Deviation')
plt.ylabel('Portfolio Expected Return')
plt.title('Random Portfolios: 3-stock vs 6-stock')
plt.legend()
plt.grid()
plt.show()


# In[ ]:


#Comment
The 3-stock and 6-stock portfolios comparison highlight the effect of diversification on the quality of the portfolio. 
The portfolios made out of the 6-stock universe have lower risks and a wider distribution of expected returns, reflect higher Sharpe ratios. 
As discussed in class, better portfolios are those that have a higher payoff per unit risk in the risk-return sense i.e. they are closer to the 
efficient frontier. This benefit is evidenced by the 6-stock portfolios which outperform better due to better diversification on a risk-adjusted basis.


# In[63]:


#Part E
# Pre-COVID expansion (2015-2019)
xiu_pre_covid = df_xiu_returns["XIU.TO"].loc["2015-01-01":"2019-12-31"]

# COVID crash + tech/commodity booms (2020-2021)
xiu_covid = df_xiu_returns["XIU.TO"].loc["2020-01-01":"2021-12-31"]

# Inflation and rate cycles (2022-2025)
xiu_inflation = df_xiu_returns["XIU.TO"].loc["2022-01-01":"2025-12-31"]


# In[64]:


#Question 1
# Pre-COVID
mean_pre_covid = xiu_pre_covid.mean()
std_pre_covid = xiu_pre_covid.std()

# COVID
mean_covid = xiu_covid.mean()
std_covid = xiu_covid.std()

# Inflation and rate cycles
mean_inflation = xiu_inflation.mean()
std_inflation = xiu_inflation.std()

print("2015-2019: Mean =", round(mean_pre_covid,4), ", Std =", round(std_pre_covid,4))
print("2020-2021: Mean =", round(mean_covid,4), ", Std =", round(std_covid,4))
print("2022-2025: Mean =", round(mean_inflation,4), ", Std =", round(std_inflation,4))


# In[65]:


#Question 2
# Pre-COVID expansion
corr_pre_covid = df_returns.loc["2015-01-01":"2019-12-31"].iloc[:, :-1].corr().mean().mean()

# COVID
corr_covid = df_returns.loc["2020-01-01":"2021-12-31"].iloc[:, :-1].corr().mean().mean()

# Inflation and rate cycles
corr_inflation = df_returns.loc["2022-01-01":"2025-12-31"].iloc[:, :-1].corr().mean().mean()

print("2015-2019 Avg Correlation:", round(corr_pre_covid,4))
print("2020-2021 Avg Correlation:", round(corr_covid,4))
print("2022-2025 Avg Correlation:", round(corr_inflation,4))


# In[66]:


#To determine the favourable regime 
mean_pre_covid, std_pre_covid = 0.007729181873810508, 0.02607347406675061
mean_covid, std_covid = 0.013756702430241444, 0.049851782341899346
mean_inflation, std_inflation = 0.010150879321452456, 0.03661779258580691

# Compute Sharpe ratios
sharpe_pre_covid = mean_pre_covid / std_pre_covid
sharpe_covid = mean_covid / std_covid
sharpe_inflation = mean_inflation / std_inflation

print("2015-2019 Sharpe Ratio:", round(sharpe_pre_covid,4))
print("2020-2021 Sharpe Ratio:", round(sharpe_covid,4))
print("2022-2025 Sharpe Ratio:", round(sharpe_inflation,4))


# In[ ]:


#How do XIU’s mean and risk vary across regimes?
•	2015-2019 (Pre COVID expansion): Mean return ≈ 0.0077, Std Dev ≈ 0.0261 - consistent but low volatility returns. 
•	2020-2021 (COVID crash): Mean return ≈ 0.0138, Std Dev ≈ 0.0499 - highest average returns, but the risk has almost double. 
•	2022-2025 (Inflation/rate cycles): Mean return ≈ 0.0102, Std Dev ≈ 0.0366 - between the two extremes, 
there is a moderate between-two-extremes mean return that is moderately risky.
    
#Does the average correlation spike during crisis periods?
•	The average correlation coefficients of the 58 stocks were minimal during the pre-COVID (0.175), 
•	Yes, it spiked during the COVID crisis (0.354), 
•	And was a little bit lower during the inflation/rate cycle (0.237).

#What does this imply about the strength of diversification?
•	Weak correlations (2015-2019): Diversification is strong - portfolios can minimize risk effectively.
•	Strong correlations (2020-2021): Diversification weakens- systemic risk takes control and portfolios will fail to avoid the fluctuations. 
•	Moderate correlations (2022-2025): The diversification has a beneficial effect, but not as much as in stable expansions.

#Which regime appears most favorable for XIU, and why?
Most favorable: 2015-2019 (Pre COVID expansion) 
Rationale: Sharpe ratio (0.296) is highest which is the best risk-adjusted performance. There was a consistent increase in returns, low volatility 
and maximum benefits of diversification. COVID had better raw returns and higher risk, whereas the inflation/rate cycle was strong and had less 
performance compared to pre-COVID.

