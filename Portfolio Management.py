#!/usr/bin/env python
# coding: utf-8

# In[105]:


import pandas as pd
import yfinance as yf
from pandas_datareader import data as pdr
import numpy as np
import math
import matplotlib.pyplot as plt
import statsmodels.api as sm


# In[27]:


#%% 1. Download and import stock price data
file_path = "Company_Student_List.xlsx"
df = pd.read_excel(file_path, sheet_name=0)

# Find row using my student ID
student_id = 400676677
my_row = df[df.iloc[:,1] == student_id]

# Extract tickers
tickers = my_row.iloc[0, 4:].dropna().tolist()

# Fix Yahoo Finance naming issue
tickers = [t.replace("BF.B", "BF-B") for t in tickers]

print("My tickers:", tickers)
print("Number of tickers:", len(tickers))

# Download data
start_date = "2005-01-01"
end_date   = "2025-12-31"

data = yf.download(
    tickers,
    start=start_date,
    end=end_date,
    interval="1d",
    auto_adjust=False,
    progress=True,
    threads=True
)

# Extract datasets
price_daily = data["Close"].copy()
adj_price_daily = data["Adj Close"].copy()
volume_daily = data["Volume"].copy()

# Ensure correct column order
price_daily = price_daily.reindex(columns=tickers)
adj_price_daily = adj_price_daily.reindex(columns=tickers)
volume_daily = volume_daily.reindex(columns=tickers)

# S&P 500 benchmark
sp500 = yf.download("^GSPC", start=start_date, end=end_date, interval="1d", auto_adjust=False)
sp500_daily = sp500[["Close", "Adj Close", "Volume"]].copy()

# Fama-French 5-Factor
ff_data = pdr.DataReader("F-F_Research_Data_5_Factors_2x3", "famafrench", start=start_date, end=end_date)
ff_5factor_monthly = ff_data[0].copy()
ff_5factor_monthly.index = ff_5factor_monthly.index.to_timestamp()

# Save to Excel
output_file = "portfolio_data.xlsx"
with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    price_daily.to_excel(writer, sheet_name="Price_daily")
    adj_price_daily.to_excel(writer, sheet_name="Adj_Price_daily")
    volume_daily.to_excel(writer, sheet_name="Volume_daily")
    sp500_daily.to_excel(writer, sheet_name="S&P 500_daily")
    ff_5factor_monthly.to_excel(writer, sheet_name="FF_5Factor_monthly")


# In[28]:


print(ff_5factor_monthly.head())
print(ff_5factor_monthly.columns)


# In[29]:


constituents = pd.read_excel(file_path, sheet_name="S&P 500 Constituent_March2025")

# Fix ticker format to match Yahoo
constituents["ticker"] = constituents["ticker"].str.replace(".", "-", regex=False)

# Filter only my assigned firms
constituents = constituents[constituents["ticker"].isin(tickers)]

# Extract data
shares_outstanding = constituents.set_index("ticker")["Share_outstanding"]
sector_map = constituents.set_index("ticker")["GICS Sector"]


# In[30]:


print(shares_outstanding.head())
print(len(shares_outstanding))


# In[31]:


print(shares_outstanding.tail())


# In[32]:


#%% 2.	Calculate Firm Information

#Market Capitalization per year
price_annual = price_daily.resample("YE").last().reindex(columns=tickers)
size_annual = price_annual.copy()

for ticker in tickers:
    size_annual[ticker] = price_annual[ticker] * shares_outstanding[ticker]

size_annual.index = size_annual.index.year


# In[33]:


size_annual.head()


# In[34]:


#Industry Market Capitalization per year
industry_size = size_annual.copy()
industry_size.columns = [sector_map[t] for t in industry_size.columns]
industry_size_annual = industry_size.T.groupby(level=0).sum().T
industry_size_annual.index.name = "Year"


# In[35]:


print(industry_size_annual.columns)


# In[36]:


# Liquidity_annual
volume_annual = volume_daily.resample("YE").sum().reindex(columns=tickers)

liquidity_annual = volume_annual.copy()

for ticker in tickers:
    liquidity_annual[ticker] = volume_annual[ticker] / shares_outstanding[ticker]

liquidity_annual.index = liquidity_annual.index.year
liquidity_annual.index.name = "Year"


# In[37]:


liquidity_annual.head()


# In[38]:


# Returns - Annual, Monthly, Daily
returns_daily = adj_price_daily.pct_change(fill_method=None).dropna(how="all")

price_monthly = adj_price_daily.resample("ME").last()
returns_monthly = price_monthly.pct_change(fill_method=None).dropna(how="all")

price_annual_adj = adj_price_daily.resample("YE").last()
returns_annual = price_annual_adj.pct_change(fill_method=None).dropna(how="all")

# Keep column order consistent
returns_daily = returns_daily.reindex(columns=tickers)
returns_monthly = returns_monthly.reindex(columns=tickers)
returns_annual = returns_annual.reindex(columns=tickers)

# Index formatting
returns_annual.index = returns_annual.index.year
returns_daily.index.name = "Date"
returns_monthly.index.name = "Date"
returns_annual.index.name = "Year"


# In[39]:


print(returns_daily.head())
print(returns_monthly.head())
print(returns_annual.head())


# In[40]:


# Risk Annual
risk_annual = returns_daily.resample("YE").std()
risk_annual.index = risk_annual.index.year
risk_annual.index.name = "Year"


# In[41]:


print(risk_annual.head())


# In[42]:


#Save all to excel
with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    price_daily.to_excel(writer, sheet_name="Price_daily")
    adj_price_daily.to_excel(writer, sheet_name="Adj_Price_daily")
    volume_daily.to_excel(writer, sheet_name="Volume_daily")
    sp500_daily.to_excel(writer, sheet_name="S&P 500_daily")
    ff_5factor_monthly.to_excel(writer, sheet_name="FF_5Factor_monthly")

    size_annual.to_excel(writer, sheet_name="Size_annual")
    industry_size_annual.to_excel(writer, sheet_name="IndustrySize_annual")
    liquidity_annual.to_excel(writer, sheet_name="Liquidity_annual")

    returns_daily.to_excel(writer, sheet_name="Returns_daily")
    returns_monthly.to_excel(writer, sheet_name="Returns_monthly")
    returns_annual.to_excel(writer, sheet_name="Returns_annual")

    risk_annual.to_excel(writer, sheet_name="Risk_annual")


# In[43]:


# #%% 3. Summary Statistics

# Range
dynamic_range = (returns_monthly.max() - returns_monthly.min()) * 12 * 100

# Median
median_return = returns_monthly.median() * 12 * 100

# Mean
mean_return = returns_monthly.mean() * 12 * 100

# Volatility
std_return = returns_monthly.std() * np.sqrt(12) * 100

# Market Capitalization
market_cap_2025 = size_annual.loc[2025]

# Industry using GIC Sector
industry_row = sector_map.reindex(tickers)


# In[28]:


print(dynamic_range.head())
print(median_return.head())
print(mean_return.head())
print(std_return.head())
print(market_cap_2025.head())
print(industry_row)


# In[46]:


# Beta Calculation

# Monthly returns for 2021-2025
returns_5year = returns_monthly.loc["2021-01-01":"2025-12-31"].copy()

# Convert returns index to monthly period so it aligns with FF data
returns_5year.index = returns_5year.index.to_period("M")

# Prepare Fama-French monthly data for same period
ff_5year = ff_5factor_monthly.loc["2021-01-01":"2025-12-31", ["Mkt-RF", "RF"]].copy()

# Convert FF data from percent to decimal
ff_5year = ff_5year / 100

# Convert FF index to monthly period too
ff_5year.index = ff_5year.index.to_period("M")

# Compute firm excess returns
excess_returns = returns_5year.sub(ff_5year["RF"], axis=0)

# Market excess return
market_excess = ff_5year["Mkt-RF"]

# Estimate CAPM beta for each firm
beta_estimated = excess_returns.apply(
    lambda col: col.cov(market_excess) / market_excess.var()
)

# March 2025 market beta from constituents sheet
beta_march_2025 = constituents.set_index("ticker")["Market Beta"].reindex(tickers)

# Percentage difference
beta_diff_pct = ((beta_estimated - beta_march_2025) / beta_march_2025) * 100


# In[47]:


print(beta_estimated.head())
print(beta_march_2025.head())
print(beta_diff_pct.head())


# In[48]:


# Combine into one DataFrame
firm_summary_stat = pd.DataFrame(
    [
        dynamic_range,
        median_return,
        mean_return,
        std_return,
        market_cap_2025,
        industry_row,
        beta_estimated,
        beta_march_2025,
        beta_diff_pct
    ],
    index=[
        "Dynamic Range (%)",
        "Median Return (%)",
        "Mean Return (%)",
        "Std Dev (%)",
        "Market Cap 2025",
        "Industry",
        "Estimated Beta",
        "March 2025 Beta",
        "Beta Difference (%)"
    ]
)

#Keep ticker order
firm_summary_stat = firm_summary_stat.reindex(columns=tickers)

# save to excel
firm_summary_stat.to_excel(writer, sheet_name="Firm_Summary_Stat")


# In[49]:


firm_summary_stat.head()


# In[50]:


firm_summary_stat.tail()


# In[51]:


# Save all to excel
with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    
    price_daily.to_excel(writer, sheet_name="Price_daily")
    adj_price_daily.to_excel(writer, sheet_name="Adj_Price_daily")
    volume_daily.to_excel(writer, sheet_name="Volume_daily")
    sp500_daily.to_excel(writer, sheet_name="S&P 500_daily")
    ff_5factor_monthly.to_excel(writer, sheet_name="FF_5Factor_monthly")

    size_annual.to_excel(writer, sheet_name="Size_annual")
    industry_size_annual.to_excel(writer, sheet_name="IndustrySize_annual")
    liquidity_annual.to_excel(writer, sheet_name="Liquidity_annual")

    returns_daily.to_excel(writer, sheet_name="Returns_daily")
    returns_monthly.to_excel(writer, sheet_name="Returns_monthly")
    returns_annual.to_excel(writer, sheet_name="Returns_annual")

    risk_annual.to_excel(writer, sheet_name="Risk_annual")

    firm_summary_stat.to_excel(writer, sheet_name="Firm_Summary_Stat")


# In[55]:


##%% 4. Construct the Portfolio

#Computing Annual Returns to make it work with MOM Strategy
returns_annual = (1 + returns_daily).groupby(returns_daily.index.year).prod() - 1
returns_annual.index.name = "Year"

# Check
print(returns_annual.head())
print(portfolio_weight_annual.head())
print(portfolio_weight_annual.sum(axis=1).head())


# In[56]:


##%% 4. Construct the Portfolio

## Portfolio Weights

# Annual portfolio weights for MOM strategy
portfolio_weight_annual = pd.DataFrame(
    0.0,
    index=range(2006, 2026),
    columns=tickers
)

for year in range(2006, 2026):
    prev_year_returns = returns_annual.loc[year - 1]

    top_10 = prev_year_returns.sort_values(ascending=False).head(10).index

    portfolio_weight_annual.loc[year, top_10] = 0.10

portfolio_weight_annual.index.name = "Year"


# In[57]:


print(portfolio_weight_annual.head())
print(portfolio_weight_annual.sum(axis=1).head())


# In[65]:


## Portfolio Monthly Returns

portfolio_return_monthly = pd.Series(
    index=returns_monthly.loc["2006-01-01":"2025-12-31"].index,
    dtype=float
)

for year in range(2006, 2026):
    yearly_months = returns_monthly.loc[f"{year}-01-01":f"{year}-12-31"]
    weights = portfolio_weight_annual.loc[year]

    portfolio_return_monthly.loc[yearly_months.index] = (
        yearly_months.mul(weights, axis=1).sum(axis=1)
    )

portfolio_return_monthly.name = "Portfolio Return"


# In[66]:


print(portfolio_return_monthly.head())
print(portfolio_return_monthly.tail())


# In[67]:


# Portfolio fund

fund_value_monthly = pd.Series(index=portfolio_return_monthly.index, dtype=float)

initial_value = 1_000_000
current_value = initial_value

for dt in portfolio_return_monthly.index:
    current_value = current_value * (1 + portfolio_return_monthly.loc[dt])
    fund_value_monthly.loc[dt] = current_value

fund_value_monthly.name = "Fund Value"


# In[68]:


print(fund_value_monthly.head())
print(fund_value_monthly.tail())


# In[70]:


# Creating adjusted price monthly needed for holdings

adj_price_monthly = adj_price_daily.resample("ME").last()
adj_price_monthly = adj_price_monthly.reindex(columns=tickers)


# In[73]:


# Porfolio hldings

portfolio_holdings_monthly = pd.DataFrame(
    index=portfolio_return_monthly.index,
    columns=tickers,
    dtype=float
)

fund_value_prev_dec = 1_000_000

for year in range(2006, 2026):
    weights = portfolio_weight_annual.loc[year].reindex(tickers)

    # Get December prices of previous year
    prev_dec_prices = adj_price_monthly.loc[
        adj_price_monthly.index.year == year - 1
    ].iloc[-1].reindex(tickers)

    # Compute shares
    holdings = (fund_value_prev_dec * weights) / prev_dec_prices

    # Assign same holdings to all months in the year
    holdings = holdings.fillna(0)

    year_months = portfolio_holdings_monthly.loc[
        f"{year}-01-01":f"{year}-12-31"
    ].index

    portfolio_holdings_monthly.loc[year_months, :] = holdings.values

    # Update fund value for next year
    fund_value_prev_dec = fund_value_monthly.loc[
        fund_value_monthly.index.year == year
    ].iloc[-1]


# In[74]:


print(portfolio_holdings_monthly.head())
print(portfolio_holdings_monthly.loc["2006-01-31"])


# In[81]:


# compute benchmark monthly return

sp500_adj_monthly = sp500_daily["Adj Close"].resample("ME").last()

benchmark_return_monthly = sp500_adj_monthly.pct_change(fill_method=None).dropna()
benchmark_return_monthly = benchmark_return_monthly.loc["2006-01-01":"2025-12-31"]

benchmark_return_monthly = benchmark_return_monthly.squeeze()

benchmark_return_monthly.name = "Benchmark Return"
benchmark_return_monthly.index.name = "Date"


# In[82]:


print(benchmark_return_monthly.head())
print(benchmark_return_monthly.tail())


# In[83]:


# Benchmark fund value

benchmark_fund_value = pd.Series(
    index=benchmark_return_monthly.index,
    dtype=float
)

benchmark_fund_value.iloc[0] = 1_000_000

for i in range(1, len(benchmark_fund_value)):
    benchmark_fund_value.iloc[i] = (
        benchmark_fund_value.iloc[i - 1]
        * (1 + benchmark_return_monthly.iloc[i])
    )

benchmark_fund_value.name = "Benchmark Fund Value"
benchmark_fund_value.index.name = "Date"


# In[78]:


print(benchmark_fund_value.head())
print(benchmark_fund_value.tail())


# In[84]:


print(fund_value_monthly.head())
print(fund_value_monthly.tail())


# In[85]:


with pd.ExcelWriter(output_file, engine="openpyxl") as writer:

    # Prices 
    price_daily.to_excel(writer, sheet_name="Price_daily")
    adj_price_daily.to_excel(writer, sheet_name="Adj_Price_daily")
    volume_daily.to_excel(writer, sheet_name="Volume_daily")
    sp500_daily.to_excel(writer, sheet_name="S&P 500_daily")
    ff_5factor_monthly.to_excel(writer, sheet_name="FF_5Factor_monthly")

    # Annual metrics
    size_annual.to_excel(writer, sheet_name="Size_annual")
    industry_size_annual.to_excel(writer, sheet_name="IndustrySize_annual")
    liquidity_annual.to_excel(writer, sheet_name="Liquidity_annual")

    # Returns
    returns_daily.to_excel(writer, sheet_name="Returns_daily")
    returns_monthly.to_excel(writer, sheet_name="Returns_monthly")
    returns_annual.to_excel(writer, sheet_name="Returns_annual")

    # Risk 
    risk_annual.to_excel(writer, sheet_name="Risk_annual")

    # Summary
    firm_summary_stat.to_excel(writer, sheet_name="Firm_Summary_Stat")


    # New Additions - Question 4

    portfolio_weight_annual.to_excel(writer, sheet_name="portfolio_weight_annual")

    portfolio_holdings_monthly.to_excel(writer, sheet_name="portfolio_holdings_monthly")

    portfolio_return_monthly.to_frame().to_excel(writer, sheet_name="PortfolioReturn_monthly")


# In[86]:


output_file = "portfolio_data.xlsx"

with pd.ExcelWriter(output_file, engine="xlsxwriter") as writer:

    #  Prices 
    price_daily.to_excel(writer, sheet_name="Price_daily")
    adj_price_daily.to_excel(writer, sheet_name="Adj_Price_daily")
    volume_daily.to_excel(writer, sheet_name="Volume_daily")
    sp500_daily.to_excel(writer, sheet_name="S&P 500_daily")
    ff_5factor_monthly.to_excel(writer, sheet_name="FF_5Factor_monthly")

    #  Annual metrics 
    size_annual.to_excel(writer, sheet_name="Size_annual")
    industry_size_annual.to_excel(writer, sheet_name="IndustrySize_annual")
    liquidity_annual.to_excel(writer, sheet_name="Liquidity_annual")

    #  Returns 
    returns_daily.to_excel(writer, sheet_name="Returns_daily")
    returns_monthly.to_excel(writer, sheet_name="Returns_monthly")
    returns_annual.to_excel(writer, sheet_name="Returns_annual")

    #  Risk 
    risk_annual.to_excel(writer, sheet_name="Risk_annual")

    #  Summary 
    firm_summary_stat.to_excel(writer, sheet_name="Firm_Summary_Stat")

    # New Additions - Question 4
    
    portfolio_weight_annual.to_excel(writer, sheet_name="portfolio_weight_annual")
    portfolio_holdings_monthly.to_excel(writer, sheet_name="portfolio_holdings_monthly")
    portfolio_return_monthly.to_frame().to_excel(writer, sheet_name="PortfolioReturn_monthly")


# In[108]:


## %% 5. Portfolio Analysis

# Portfolio stats (annualized %)
portfolio_stats = {
    "Mean (%)": portfolio_return_monthly.mean() * 12 * 100,
    "Median (%)": portfolio_return_monthly.median() * 12 * 100,
    "Std (%)": portfolio_return_monthly.std() * (12**0.5) * 100,
    "Dynamic Range (%)": (portfolio_return_monthly.max() - portfolio_return_monthly.min()) * 12 * 100
}

# Benchmark stats (annualized %)
benchmark_stats = {
    "Mean (%)": benchmark_return_monthly.mean() * 12 * 100,
    "Median (%)": benchmark_return_monthly.median() * 12 * 100,
    "Std (%)": benchmark_return_monthly.std() * (12**0.5) * 100,
    "Dynamic Range (%)": (benchmark_return_monthly.max() - benchmark_return_monthly.min()) * 12 * 100
}


# In[111]:


# Align data properly
portfolio_reg = portfolio_return_monthly.copy()
benchmark_reg = benchmark_return_monthly.copy()

portfolio_reg.index = portfolio_reg.index.to_period("M")
benchmark_reg.index = benchmark_reg.index.to_period("M")

ff = ff_5factor_monthly.copy() / 100
ff = ff.loc["2006-01-01":"2025-12-31"]
ff.index = ff.index.to_period("M")

# Combine and clean
portfolio_data = pd.concat([portfolio_reg, ff], axis=1).dropna()
benchmark_data = pd.concat([benchmark_reg, ff], axis=1).dropna()

# Excess returns
portfolio_excess = portfolio_data.iloc[:, 0] - portfolio_data["RF"]
benchmark_excess = benchmark_data.iloc[:, 0] - benchmark_data["RF"]

# Factors
X_port = sm.add_constant(portfolio_data[["Mkt-RF","SMB","HML","RMW","CMA"]])
X_bench = sm.add_constant(benchmark_data[["Mkt-RF","SMB","HML","RMW","CMA"]])

# Regression
model_portfolio = sm.OLS(portfolio_excess, X_port).fit()
model_benchmark = sm.OLS(benchmark_excess, X_bench).fit()

# Alpha (annual %)
alpha_portfolio_annual = model_portfolio.params["const"] * 12 * 100
alpha_benchmark_annual = model_benchmark.params["const"] * 12 * 100


# In[112]:


fund_summary.loc["Alpha (%)"] = [
    alpha_portfolio_annual,
    alpha_benchmark_annual
]


# In[113]:


# Average monthly risk-free rate
rf_mean = ff["RF"].mean()

# Sharpe Ratio
sharpe_portfolio = ((portfolio_return_monthly.mean() - rf_mean) / portfolio_return_monthly.std()) * (12**0.5)
sharpe_benchmark = ((benchmark_return_monthly.mean() - rf_mean) / benchmark_return_monthly.std()) * (12**0.5)

fund_summary.loc["Sharpe Ratio"] = [sharpe_portfolio, sharpe_benchmark]

# Treynor Ratio
beta_portfolio = model_portfolio.params["Mkt-RF"]
beta_benchmark = model_benchmark.params["Mkt-RF"]

treynor_portfolio = ((portfolio_return_monthly.mean() - rf_mean) * 12) / beta_portfolio
treynor_benchmark = ((benchmark_return_monthly.mean() - rf_mean) * 12) / beta_benchmark

fund_summary.loc["Treynor Ratio"] = [treynor_portfolio, treynor_benchmark]

# Round to 4 decimal places
fund_summary = fund_summary.round(4)
print(fund_summary)


# In[119]:


# Final holdings and prices

final_holdings = portfolio_holdings_monthly.loc["2025-12-31"]
final_prices = adj_price_monthly.loc["2025-12-31"]

# Compute final market values

final_market_value = final_holdings * final_prices

#convert to portfolio percentages

funds_holdings_composition = (final_market_value / final_market_value.sum()) * 100

# Clean table for excel purpose
print(funds_holdings_composition)
print(funds_holdings_composition["Weight (%)"].sum())


# In[120]:


print(funds_holdings_composition)
print(funds_holdings_composition["Weight (%)"].sum())


# In[124]:


print(funds_holdings_composition.head())


# In[122]:


output_file = "portfolio_data.xlsx"

with pd.ExcelWriter(output_file, engine="xlsxwriter") as writer:

    # Prices 
    price_daily.to_excel(writer, sheet_name="Price_daily")
    adj_price_daily.to_excel(writer, sheet_name="Adj_Price_daily")
    volume_daily.to_excel(writer, sheet_name="Volume_daily")
    sp500_daily.to_excel(writer, sheet_name="S&P 500_daily")
    ff_5factor_monthly.to_excel(writer, sheet_name="FF_5Factor_monthly")

    # Annual metrics 
    size_annual.to_excel(writer, sheet_name="Size_annual")
    industry_size_annual.to_excel(writer, sheet_name="IndustrySize_annual")
    liquidity_annual.to_excel(writer, sheet_name="Liquidity_annual")

    # Returns 
    returns_daily.to_excel(writer, sheet_name="Returns_daily")
    returns_monthly.to_excel(writer, sheet_name="Returns_monthly")
    returns_annual.to_excel(writer, sheet_name="Returns_annual")

    # Risk 
    risk_annual.to_excel(writer, sheet_name="Risk_annual")

    # Summary 
    firm_summary_stat.to_excel(writer, sheet_name="Firm_Summary_Stat")

    # New additions Q4 
    portfolio_weight_annual.to_excel(writer, sheet_name="portfolio_weight_annual")
    portfolio_holdings_monthly.to_excel(writer, sheet_name="portfolio_holdings_monthly")
    portfolio_return_monthly.to_frame(name="Portfolio Return").to_excel(
        writer, sheet_name="PortfolioReturn_monthly"
    )

    # Add Fund composition - Q5
    fund_summary.to_excel(writer, sheet_name="Fund_summary")
    funds_holdings_composition.to_excel(writer, sheet_name="Funds_Holdings_Composition")


# In[125]:


# Add sector information
composition_industry = funds_holdings_composition.copy()
composition_industry["Industry"] = composition_industry.index.map(sector_map)

# sum weights by industry
industry_composition = composition_industry.groupby("Industry")["Weight (%)"].sum()

print(industry_composition)
print(industry_composition.sum())


# In[126]:


# Plot Pie - Fund Industry Composition

plt.figure(figsize=(8, 8))
industry_composition.plot(kind="pie", autopct="%1.1f%%")

plt.ylabel("")
plt.title("Fund Industry Composition (Dec 2025)")
plt.tight_layout()
plt.savefig("Fund_composition_Industry.pdf")
plt.show()


# In[127]:


# Histogram plot for monthly return
plt.figure()

portfolio_return_monthly.plot(
    kind="hist",
    bins=20
)

plt.title("Distribution of Fund Monthly Returns")
plt.xlabel("Monthly Return")
plt.ylabel("Frequency")

plt.tight_layout()
plt.savefig("Fund_return_hist.pdf")

plt.show()


# In[128]:


# import matplotlib.pyplot as plt

plt.figure()

fund_value_monthly.plot(label="Portfolio")
benchmark_fund_value.plot(label="Benchmark")

plt.title("Fund Value: Portfolio vs Benchmark")
plt.xlabel("Date")
plt.ylabel("Fund Value ($)")

plt.legend()
plt.tight_layout()
plt.savefig("Fund_value.pdf")

plt.show()

plt.figure()

fund_value_monthly.plot(label="Portfolio")
benchmark_fund_value.plot(label="Benchmark")

plt.title("Fund Value: Portfolio vs Benchmark")
plt.xlabel("Date")
plt.ylabel("Fund Value ($)")

plt.legend()
plt.tight_layout()
plt.savefig("Fund_value.pdf")

plt.show()


# In[130]:


# Sum of monthly returns per year 

# Annual returns from monthly returns
portfolio_annual_return = portfolio_return_monthly.groupby(
    portfolio_return_monthly.index.year
).sum()

benchmark_annual_return = benchmark_return_monthly.groupby(
    benchmark_return_monthly.index.year
).sum()

# Combine into one table
annual_returns_df = pd.DataFrame({
    "Portfolio": portfolio_annual_return,
    "Benchmark": benchmark_annual_return
})

# Bar chart
ax = annual_returns_df.plot(kind="bar")

ax.set_title("Annual Returns: Portfolio vs Benchmark")
ax.set_xlabel("Year")
ax.set_ylabel("Annual Return")

plt.tight_layout()
plt.savefig("Fund_return_annual.pdf")
plt.show()


# In[ ]:




