#!/usr/bin/env python
# coding: utf-8

# In[18]:


#%% Question 1
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[21]:


price = 1_400_000
down_payment_rate = 0.20
years = 20
apr = 0.0529

## Inefficiences in the original code:
# It created extra variables like:
# down_payment = 0.20 * price
# loan = price - down_payment
# r = 0.0529
# n = years
# Some of these aliases are unnecessary and make the code less direct.

loan = price * (1 - down_payment_rate)


# In[22]:


#Compute annual payment

# Mortgage payment formula:
# PMT = L * [r(1+r)^n] / [(1+r)^n - 1]

payment = loan * (apr * (1+apr)**years) / ((1+apr)**years - 1)


# In[23]:


# Build Ammortization Schedule

# Original code inefficiency:
# It used many empty Python lists and appended inside the loop.
# That works, but since we already know the number of years,
# preallocating NumPy arrays is cleaner and more efficient.

year = np.arange(1, years + 1)
total_payment = np.full(years, payment)
interest_paid = np.zeros(years)
principal_paid = np.zeros(years)
end_balance = np.zeros(years)

balance = loan

# A loop is still needed here because each year's balance depends
# on the previous year's balance. So this part is naturally recursive.

for t in range(years):
    interest_paid[t] = balance * apr
    principal_paid[t] = payment - interest_paid[t]
    balance -= principal_paid[t]
    
    # Original code issue:
    # It did not handle floating-point residue at the end,
    # so the final balance may show as a tiny negative number.
    if t == years - 1:
        balance = 0

    end_balance[t] = balance


# In[25]:


# Convert to DataFrame

# Original code inefficiencies:
# Converting to DataFrame only after building several Python lists is okay,
# but using arrays from the start is more efficient and cleaner.


df = pd.DataFrame({
    "Year": year,
    "Total_Payment": total_payment,
    "Interest_Paid": interest_paid,
    "Principal_Paid": principal_paid,
    "End_Balance": end_balance
})  
print(df)


# In[31]:


# Plot

# Inefficiency in original code:
# The original answer plotted all variables as line graphs.
# That does not match the required output shown in the question.

# Inefficiency in original code:
# Plotting Total Payment as a separate line is unnecessary here,
# because Total Payment = Interest Paid + Principal Paid.
# A stacked bar chart already shows the total payment and its breakdown.

# Inefficiency in original code:
# The original plot used one axis style for variables with different visual purposes.
# The remaining loan balance is better shown as a line on a secondary axis.

# Inefficiency in original code:
# The original code also had inconsistent column naming between the DataFrame
# and the plot code, which caused KeyError issues.

fig, ax1 = plt.subplots(figsize=(10, 6))

# stacked bars for payment components
ax1.bar(df["Year"], df["Interest_Paid"], label="Interest Paid")
ax1.bar(df["Year"], df["Principal_Paid"],
        bottom=df["Interest_Paid"], label="Principal Paid")

ax1.set_xlabel("Year")
ax1.set_ylabel("Annual Payment ($)")
ax1.set_title("Amortization Table")
ax1.legend(loc="upper left")
ax1.grid(axis="y", alpha=0.3)

# line for remaining balance on secondary axis
ax2 = ax1.twinx()
ax2.plot(df["Year"], df["End_Balance"], label="Loan Balance")
ax2.legend(loc="upper left")
ax1.legend(loc="lower center")
plt.show()


# In[32]:


#%% Question 2

def anticipated_depreciation(car_value, term_years, monthly_payment, annual_rate):
    """
    Returns residual value, retention rate, and depreciation rate
    implied by a lease contract.
    """
    n = term_years * 12         # total number of months
    r_m = annual_rate / 12      # monthly interest rate

    # Present value of lease payments
    pv_payments = monthly_payment * (1 - (1 + r_m) ** (-n)) / r_m

    # Residual value implied by the lease
    residual_value = (car_value - pv_payments) * (1 + r_m) ** n

    # Ratios
    retention_rate = residual_value / car_value
    depreciation_rate = 1 - retention_rate

    return residual_value, retention_rate, depreciation_rate


# In[37]:


# BMW 330i xDrive Sedan
bmw_residual, bmw_retention, bmw_depreciation = anticipated_depreciation(
    car_value=74000,
    term_years=5,
    monthly_payment=927.53,
    annual_rate=0.0299
)

# Mercedes-Benz C 300 Sedan
merc_residual, merc_retention, merc_depreciation = anticipated_depreciation(
    car_value=75360,
    term_years=5,
    monthly_payment=888.09,
    annual_rate=0.0319
)

print("For BMW:")
print("BMW residual value:", round(bmw_residual, 2))
print("BMW retention rate:", round(bmw_retention, 4))
print("BMW depreciation rate:", round(bmw_depreciation, 4))

print("For Mercedes:")
print("Mercedes residual value:", round(merc_residual, 2))
print("Mercedes retention rate:", round(merc_retention, 4))
print("Mercedes depreciation rate:", round(merc_depreciation, 4))


# In[ ]:


#The BMW’s implied depreciation rate is about 64.91%, while the Mercedes-Benz’s is about 59.28%. 
#Based on the above results, the Mercedes-Benz C 300 Sedan is expected to depreciate less and retain a higher proportion of its value after 5 years


# In[38]:


#%% Question 3

# Input data

current_age = 22
retire_age = 64
last_age = 94

current_expense = 45000
inflation = 0.02
pre_ret_return = 0.11
ret_return = 0.04
tax_rate = 0.30
volatility = 0.16

# Clarification-based timing
deposit_ages = list(range(23, 65))      # 23 to 64 inclusive
withdrawal_ages = list(range(65, 95))   # 65 to 94 inclusive

n_deposits = len(deposit_ages)          # 42
n_withdrawals = len(withdrawal_ages)    # 30


# In[39]:


# Part a - Find required retirement fund

# First retirement year's after-tax expense (age 65 consumption)
years_to_first_retirement_spending = 65 - 22
first_after_tax_retirement_expense = current_expense * (1 + inflation) ** years_to_first_retirement_spending

# Gross withdrawal needed from RRSP because withdrawals are taxed
first_gross_withdrawal = first_after_tax_retirement_expense / (1 - tax_rate)

# Present value at retirement of a growing annuity due
# PV_due = W1 * [(1 - ((1+g)/(1+r))^n) / (r-g)] * (1+r)
retirement_goal = (
    first_gross_withdrawal
    * (1 - ((1 + inflation) / (1 + ret_return)) ** n_withdrawals)
    / (ret_return - inflation)
    * (1 + ret_return)
)

# Required annual RRSP contribution
annual_deposit = retirement_goal / (((1 + pre_ret_return) ** n_deposits - 1) / pre_ret_return)

print("First after-tax retirement expense:", round(first_after_tax_retirement_expense, 2))
print("First gross withdrawal from RRSP:", round(first_gross_withdrawal, 2))
print("Retirement goal at age 64:", round(retirement_goal, 2))
print("Required annual RRSP deposit:", round(annual_deposit, 2))


# In[40]:


# Build deterministic schedule
ages = list(range(22, 95))
df = pd.DataFrame({"Age": ages})

df["Deposit"] = 0.0
df["Withdrawal"] = 0.0
df["Balance_End"] = 0.0

balance = 0.0

for age in ages:
    # retirement years: withdrawal at beginning of year
    if age in withdrawal_ages:
        k = age - 65   # 0 for age 65, 1 for age 66, ...
        after_tax_expense = current_expense * (1 + inflation) ** (age - 22)
        gross_withdrawal = after_tax_expense / (1 - tax_rate)
        df.loc[df["Age"] == age, "Withdrawal"] = gross_withdrawal
        balance -= gross_withdrawal

    # growth during year
    if age < 64:
        balance *= (1 + pre_ret_return)
    elif age >= 64 and age < 94:
        balance *= (1 + ret_return)
    elif age == 94:
        balance *= (1 + ret_return)

    # saving years: deposit at end of year
    if age in deposit_ages:
        df.loc[df["Age"] == age, "Deposit"] = annual_deposit
        balance += annual_deposit

    df.loc[df["Age"] == age, "Balance_End"] = balance

# small rounding cleanup
df.loc[df["Age"] == 94, "Balance_End"] = 0.0

print(df.head(10))
print(df.tail(10))


# In[41]:


# Part b - Run 100 times simulations

n_sims = 100
simulated_balances = np.zeros((len(ages), n_sims))

for sim in range(n_sims):
    balance = 0.0
    yearly_balances = []

    for i, age in enumerate(ages):
        # withdrawal at beginning of year during retirement
        if age in withdrawal_ages:
            gross_withdrawal = current_expense * (1 + inflation) ** (age - 22) / (1 - tax_rate)

            if balance >= gross_withdrawal:
                balance -= gross_withdrawal
            else:
                # consume everything, so balance cannot go negative
                balance = 0.0

        # apply return during year
        if age < 64:
            realized_return = np.random.normal(pre_ret_return, volatility)
            balance *= (1 + realized_return)
        else:
            balance *= (1 + ret_return)

        # deposit at end of year during saving period
        if age in deposit_ages:
            balance += annual_deposit

        yearly_balances.append(balance)

    simulated_balances[:, sim] = yearly_balances

# Save one simulation and average simulation
df["Simulated_Balance_1"] = simulated_balances[:, 0]
df["Average_Simulated_Balance"] = simulated_balances.mean(axis=1)

print(df.head(10))
print(df.tail(10))


# In[72]:


[
 'Age',
 'Deposit',
 'Withdrawal',
 'End of year balance',
 'Simulated Balance',
 'Average Simulated Balance'
]


# In[73]:


df_output = df_output[
    [
        "Age",
        "Deposit",
        "Withdrawal",
        "End of year balance",
        "Simulated Balance",
        "Average Simulated Balance"
    ]
]


# In[74]:


df_output = df_output.round(2)


# In[75]:


file_path = "Input_Data_for_Final_Exam.xlsx"

with pd.ExcelWriter(file_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
    df_output.to_excel(writer, sheet_name="Q3-Output", index=False)


# In[ ]:





# In[79]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import statsmodels.api as sm

# =========================================================
# FILE PATH
# =========================================================
file_path = "Input_Data_for_Final_Exam.xlsx"

# =========================================================
# STOCK SHEETS
# =========================================================
stock_sheets = ["NVDA", "NFLX", "MCD", "GS", "AMZN", "SBUX", "CRM"]

# =========================================================
# HELPER FUNCTION: PREPARE MONTHLY PRICES AND RETURNS
# =========================================================
def prepare_stock_monthly(sheet_name, file_path):
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    df.columns = [str(c).strip() for c in df.columns]

    # Your workbook uses "Price" as the date column
    if "Price" in df.columns:
        date_col = "Price"
    else:
        date_col = df.columns[0]

    # Find Adj Close column
    adj_col = None
    for c in df.columns:
        cl = c.lower().replace(" ", "")
        if "adjclose" in cl or "adjustedclose" in cl:
            adj_col = c
            break

    # Fallback to Close
    if adj_col is None:
        for c in df.columns:
            if "close" in c.lower():
                adj_col = c
                break

    if adj_col is None:
        raise ValueError(f"Could not find Close/Adj Close column in sheet {sheet_name}")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[adj_col] = pd.to_numeric(df[adj_col], errors="coerce")

    df = df[[date_col, adj_col]].dropna().sort_values(date_col)
    df = df.rename(columns={date_col: "Date", adj_col: sheet_name})
    df = df.set_index("Date")

    # Month-end frequency: use ME
    monthly_price = df.resample("ME").last()
    monthly_ret = monthly_price.pct_change()

    return monthly_price, monthly_ret

# =========================================================
# PART (a): MONTHLY STOCK RETURNS
# =========================================================
monthly_prices_dict = {}
monthly_returns_list = []

for s in stock_sheets:
    px, ret = prepare_stock_monthly(s, file_path)
    monthly_prices_dict[s] = px
    monthly_returns_list.append(ret)

monthly_prices = pd.concat([monthly_prices_dict[s] for s in stock_sheets], axis=1, join="outer")
monthly_returns = pd.concat(monthly_returns_list, axis=1, join="outer")

monthly_prices = monthly_prices.sort_index()
monthly_returns = monthly_returns.sort_index()

# =========================================================
# READ FAMA-FRENCH FACTORS
# =========================================================
# Your FF sheet already has usable headers in row 0
ff = pd.read_excel(file_path, sheet_name="FF_Factors_FromKennethFrenchWeb")
ff.columns = [str(c).strip() for c in ff.columns]

# Convert YYYY-MM to month-end date
ff["Date"] = pd.to_datetime(ff["Date"], format="%Y-%m", errors="coerce") + pd.offsets.MonthEnd(0)

needed_ff = ["Date", "Mkt-RF", "RF", "SMB", "HML"]
ff = ff[needed_ff].copy()

for c in ["Mkt-RF", "RF", "SMB", "HML"]:
    ff[c] = pd.to_numeric(ff[c], errors="coerce") / 100.0

ff = ff.dropna().set_index("Date").sort_index()

# Merge with all observations
monthly_data = monthly_returns.merge(ff, left_index=True, right_index=True, how="outer")
monthly_data = monthly_data.sort_index()

# =========================================================
# PART (b): CONSTRUCT TWO FUNDS
# =========================================================
rebalance_dates = pd.date_range("2011-01-31", "2025-12-31", freq="ME")
value_dates = pd.date_range("2010-12-31", "2025-12-31", freq="ME")

initial_endowment = 1_000_000

stock_return_data = monthly_data[stock_sheets].copy()
stock_return_data = stock_return_data.loc[
    (stock_return_data.index >= pd.Timestamp("2006-01-31")) &
    (stock_return_data.index <= pd.Timestamp("2025-12-31"))
]

ff_sub = ff.loc[
    (ff.index >= pd.Timestamp("2006-01-31")) &
    (ff.index <= pd.Timestamp("2025-12-31"))
]

def neg_sharpe(weights, mean_ret, cov_mat, rf):
    port_ret = np.dot(weights, mean_ret)
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_mat, weights)))
    if port_vol == 0:
        return 1e6
    return -((port_ret - rf) / port_vol)

def get_sharpe_weights(hist_returns, rf):
    hist_returns = hist_returns.dropna(how="any")
    n = hist_returns.shape[1]

    if len(hist_returns) < 12:
        return np.repeat(1/n, n)

    mean_ret = hist_returns.mean().values
    cov_mat = hist_returns.cov().values

    x0 = np.repeat(1/n, n)
    bounds = [(0, 1)] * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

    result = minimize(
        neg_sharpe,
        x0=x0,
        args=(mean_ret, cov_mat, rf),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )

    if result.success:
        w = result.x
    else:
        w = x0

    return w / w.sum()

def get_random_weights(n):
    w = np.random.random(n)
    return w / w.sum()

fund_values = pd.DataFrame(
    index=value_dates,
    columns=["Fund_SharpeRatio_Value", "Fund_Random_Value"],
    dtype=float
)
fund_values.loc[pd.Timestamp("2010-12-31"), "Fund_SharpeRatio_Value"] = initial_endowment
fund_values.loc[pd.Timestamp("2010-12-31"), "Fund_Random_Value"] = initial_endowment

share_records = []

current_value_sharpe = initial_endowment
current_value_random = initial_endowment

for current_date in rebalance_dates:
    prev_date = current_date - pd.offsets.MonthEnd(1)

    if prev_date not in monthly_prices.index or current_date not in monthly_prices.index:
        continue

    start_prices = monthly_prices.loc[prev_date, stock_sheets]
    end_prices = monthly_prices.loc[current_date, stock_sheets]

    if start_prices.isna().any() or end_prices.isna().any():
        continue

    hist_start = prev_date - pd.DateOffset(years=5)
    hist_returns = stock_return_data.loc[
        (stock_return_data.index > hist_start) &
        (stock_return_data.index <= prev_date),
        stock_sheets
    ]

    rf_val = ff_sub["RF"].mean()
    if prev_date in ff_sub.index:
        rf_val = ff_sub.loc[prev_date, "RF"]

    # Sharpe fund
    sharpe_weights = get_sharpe_weights(hist_returns, rf_val)
    sharpe_alloc = current_value_sharpe * sharpe_weights
    sharpe_shares = sharpe_alloc / start_prices.values
    sharpe_end_value = np.sum(sharpe_shares * end_prices.values)

    fund_values.loc[current_date, "Fund_SharpeRatio_Value"] = sharpe_end_value

    sharpe_record = {"Date": current_date, "Fund": "Fund_SharpeRatio"}
    for i, stock in enumerate(stock_sheets):
        sharpe_record[f"{stock}_Shares"] = sharpe_shares[i]
    share_records.append(sharpe_record)

    current_value_sharpe = sharpe_end_value

    # Random fund
    random_weights = get_random_weights(len(stock_sheets))
    random_alloc = current_value_random * random_weights
    random_shares = random_alloc / start_prices.values
    random_end_value = np.sum(random_shares * end_prices.values)

    fund_values.loc[current_date, "Fund_Random_Value"] = random_end_value

    random_record = {"Date": current_date, "Fund": "Fund_Random"}
    for i, stock in enumerate(stock_sheets):
        random_record[f"{stock}_Shares"] = random_shares[i]
    share_records.append(random_record)

    current_value_random = random_end_value

fund_values = fund_values.sort_index()
share_holdings = pd.DataFrame(share_records).sort_values(["Date", "Fund"])

# =========================================================
# PART (c): PLOT TOTAL FUND VALUES AND SAVE AS PDF
# =========================================================
plt.figure(figsize=(10, 6))
plt.plot(fund_values.index, fund_values["Fund_SharpeRatio_Value"], label="Fund_SharpeRatio")
plt.plot(fund_values.index, fund_values["Fund_Random_Value"], label="Fund_Random")
plt.xlabel("Date")
plt.ylabel("Fund Value ($)")
plt.title("Total Value of Both Funds")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("Q4_Fund_Value_Plot.pdf")
plt.show()

# =========================================================
# PART (d): MONTHLY FUND RETURNS
# =========================================================
fund_returns = fund_values.pct_change().loc["2011-01-31":"2025-12-31"].copy()
fund_returns.columns = ["Fund_SharpeRatio_Return", "Fund_Random_Return"]

# =========================================================
# PART (e): PERFORMANCE METRICS
# =========================================================
def calc_metrics(fund_ret_series, ff_data):
    temp = pd.concat([fund_ret_series, ff_data[["Mkt-RF", "RF", "SMB", "HML"]]], axis=1, join="inner").dropna()
    temp.columns = ["FundRet", "Mkt-RF", "RF", "SMB", "HML"]

    temp["ExcessFund"] = temp["FundRet"] - temp["RF"]

    n_months = len(temp)
    cgar = (1 + temp["FundRet"]).prod() ** (12 / n_months) - 1 if n_months > 0 else np.nan

    fund_std = temp["FundRet"].std()
    sharpe = (temp["ExcessFund"].mean() / fund_std) * np.sqrt(12) if fund_std != 0 else np.nan

    X = temp[["Mkt-RF", "SMB", "HML"]]
    X = sm.add_constant(X)
    y = temp["ExcessFund"]

    model = sm.OLS(y, X).fit()

    alpha_monthly = model.params["const"]
    beta = model.params["Mkt-RF"]

    alpha_annual = alpha_monthly * 12
    mean_excess_annual = temp["ExcessFund"].mean() * 12
    treynor = mean_excess_annual / beta if beta != 0 else np.nan

    return {
        "CGAR": cgar,
        "Alpha": alpha_annual,
        "Beta": beta,
        "Sharpe Ratio": sharpe,
        "Treynor Ratio": treynor
    }

metrics_sharpe = calc_metrics(fund_returns["Fund_SharpeRatio_Return"], ff_sub)
metrics_random = calc_metrics(fund_returns["Fund_Random_Return"], ff_sub)

performance_metrics = pd.DataFrame([
    {"Fund": "Fund_SharpeRatio", **metrics_sharpe},
    {"Fund": "Fund_Random", **metrics_random}
])

# =========================================================
# SAVE ALL Q4 OUTPUTS
# =========================================================
with pd.ExcelWriter(file_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
    monthly_data.reset_index().to_excel(writer, sheet_name="Q4-MonthlyReturns", index=False)
    share_holdings.to_excel(writer, sheet_name="Q4-ShareHoldings", index=False)
    fund_values.reset_index().to_excel(writer, sheet_name="Q4-FundValues", index=False)
    fund_returns.reset_index().to_excel(writer, sheet_name="Q4-FundReturns", index=False)
    performance_metrics.to_excel(writer, sheet_name="Q4-PerformanceMetrics", index=False)


# In[77]:


print(pd.read_excel(file_path, sheet_name="NVDA").columns.tolist())
print(pd.read_excel(file_path, sheet_name="FF_Factors_FromKennethFrenchWeb", header=None).head(15))


# In[ ]:





# In[ ]:




