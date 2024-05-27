import pandas as pd
import numpy as np
from fredapi import Fred
from statsmodels.tsa.filters.hp_filter import hpfilter
import matplotlib.pyplot as plt
import seaborn as sns


# authors:
# Hadia Eshanzada
# Fatemeh Amirian
# Hossein Yousefian

api_key = '77f214f1344420c3169b2176bb36d21a'
fred = Fred(api_key=api_key)

variables = {
    'GDP': 'GDP',                   # Gross Domestic Product
    'CND': 'PCEND',              # Personal Consumption Expenditures: Nondurable Goods
    'CD': 'PCEDG',               # Personal Consumption Expenditures: Durable Goods
    'H': 'HOHWMN02USM065S',         # Hours: Hours Worked: Manufacturing: Weekly for United States
    'AveH': 'AWHMAN',               # Average Weekly Hours of Production and Nonsupervisory Employees: Manufacturing
    'L': 'PAYEMS',                  # All Employees: Total Nonfarm
    'AveW': 'CES3000000008'         # Average Hourly Earnings of Production and Nonsupervisory Employees, Manufacturing
}

table_one_data = {var: fred.get_series(series_id) for var, series_id in variables.items()}
table_one_df = pd.DataFrame(table_one_data)
table_one_df = table_one_df.resample('Q').mean()
table_one_df.dropna(inplace=True)
table_one_df['GDP/L'] = table_one_df['GDP'] / table_one_df['L']
table_one_df.to_csv('table_one_final.csv')

table_one_df_pct_change = table_one_df.pct_change().dropna()
std_devs = table_one_df_pct_change.std() * 100
lags = list(range(-4, 5))
cross_correlations = {}

for column in table_one_df_pct_change.columns:
    cross_correlations[column] = [table_one_df_pct_change['GDP'].shift(-lag).corr(table_one_df_pct_change[column]) for lag in lags]

cross_corr_df = pd.DataFrame(cross_correlations, index=[f't{lag:+d}' for lag in lags])
cross_corr_df.loc['SD%'] = std_devs
cross_corr_df = cross_corr_df.T
print(cross_corr_df)

tex_table_one = cross_corr_df.to_latex(float_format="%.2f")

with open('table_one.tex', 'w') as f:
    f.write(tex_table_one)

corr_matrix = table_one_df_pct_change.corr()
plt.figure(figsize=(12, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlations of Economic Indicators')
plt.savefig('correlation_table_one.png')
plt.show()


""" Line Charts for Key Variables """
plt.figure(figsize=(10, 6))
plt.plot(table_one_df.index, table_one_df['GDP'], label='GDP')
plt.plot(table_one_df.index, table_one_df['CND'], label='Non-Durable Consumption')
plt.plot(table_one_df.index, table_one_df['CD'], label='Durable Consumption')
plt.xlabel('Date')
plt.ylabel('Logarithmic Value')
plt.yscale('log')
plt.title('GDP and Consumption Trends')
plt.legend()
plt.grid(True)
plt.savefig('GDP_and_Consumption_Trends.png')
plt.show()


plt.figure(figsize=(10, 6))
plt.plot(table_one_df.index, table_one_df['H'], label='Total Hours Worked')
plt.plot(table_one_df.index, table_one_df['AveH'], label='Average Hours Worked')
plt.plot(table_one_df.index, table_one_df['L'], label='Employment')
plt.xlabel('Date')
plt.ylabel('Logarithmic Value')
plt.yscale('log')
plt.title('Labor Market Trends')
plt.legend()
plt.grid(True)
plt.savefig('Labor_Market_Trends.png')
plt.show()


plt.figure(figsize=(10, 6))
plt.plot(table_one_df.index, table_one_df['AveW'], label='Average Wage')
plt.plot(table_one_df.index, table_one_df['GDP/L'], label='Productivity')
plt.xlabel('Date')
plt.ylabel('Logarithmic Value')
plt.yscale('log')
plt.title('Wages and Productivity Trends')
plt.legend()
plt.grid(True)
plt.savefig('Wages_and_Productivity_Trends.png')
plt.show()


variables = {
    'Y': 'A939RX0Q048SBEA',                   # Real gross domestic product per capita
    'C': 'A794RX0Q048SBEA',              # Real personal consumption expenditures per capita
    'IT': 'GPDIC1',               # Real Gross Private Domestic Investment
    'P': 'POPTHM',                  # Population
    'N': 'HOHWMN02USM065S',         # Hours: Hours Worked: Manufacturing: Weekly for United States
    'w': 'COMPRNFB',                  # Nonfarm Business Sector: Real Hourly Compensation for All Workers
    'FR': 'FEDFUNDS',        # Federal Funds Effective Rate
    'CPI': 'CPALTT01USM657N',   # Consumer Price Index: All Items: Total for United States
    # 'A': 'MFPNFBS'         # Private Nonfarm Business Sector: Total Factor Productivity
}

table_two_data = {var: fred.get_series(series_id) for var, series_id in variables.items()}
table_two_df_q = pd.DataFrame(table_two_data)
table_two_df_q = table_two_df_q.resample('Q').mean()
table_two_df_q.dropna(inplace=True)
table_two_df_q['I'] = table_two_df_q['IT'] / table_two_df_q['P']     # Investment per capita
table_two_df_q['r'] = table_two_df_q['FR'] - table_two_df_q['CPI']    # Real interest rate
table_two_df_q['Y/N'] = table_two_df_q['Y'] - table_two_df_q['N']
table_two_df_q = table_two_df_q.drop(columns=['IT', 'P', 'CPI', 'FR'])
table_two_df_q.to_csv('table_two_q_final.csv')

for var in ['Y', 'C', 'I', 'N', 'Y/N', 'w']:
    table_two_df_q[var] = np.log(table_two_df_q[var])

for column in table_two_df_q.columns:
    cycle, _ = hpfilter(table_two_df_q[column], lamb=1600)
    table_two_df_q[column] = cycle

stats = pd.DataFrame({
    'Variable': ['Y', 'C', 'I', 'N', 'Y/N', 'w', 'r'],
    'SD': [table_two_df_q[var].std() for var in ['Y', 'C', 'I', 'N', 'Y/N', 'w', 'r']],
    'Relative SD': [table_two_df_q[var].std() / table_two_df_q['Y'].std() for var in ['Y', 'C', 'I', 'N', 'Y/N', 'w', 'r']],
    'ρ': [table_two_df_q[var].autocorr() for var in ['Y', 'C', 'I', 'N', 'Y/N', 'w', 'r']],
    'Corr(·, Y)': [table_two_df_q[var].corr(table_two_df_q['Y']) for var in ['Y', 'C', 'I', 'N', 'Y/N', 'w', 'r']]
})
latex_code = stats.to_latex(index=False, float_format="%.2f")

with open('table_two.tex', 'w') as f:
    f.write(latex_code)


variables = {
    'Y': 'A939RX0Q048SBEA',                   # Real gross domestic product per capita
    'C': 'A794RX0Q048SBEA',              # Real personal consumption expenditures per capita
    'IT': 'GPDIC1',               # Real Gross Private Domestic Investment
    'P': 'POPTHM',                  # Population
    'N': 'HOHWMN02USM065S',         # Hours: Hours Worked: Manufacturing: Weekly for United States
    'w': 'COMPRNFB',                  # Nonfarm Business Sector: Real Hourly Compensation for All Workers
    'FR': 'FEDFUNDS',        # Federal Funds Effective Rate
    'CPI': 'CPALTT01USM657N',   # Consumer Price Index: All Items: Total for United States
    'A': 'MFPNFBS'         # Private Nonfarm Business Sector: Total Factor Productivity (yearly)
}

table_two_data = {var: fred.get_series(series_id) for var, series_id in variables.items()}
table_two_df_y = pd.DataFrame(table_two_data)
table_two_df_y = table_two_df_y.resample('Y').mean()
table_two_df_y.dropna(inplace=True)
table_two_df_y['I'] = table_two_df_y['IT'] / table_two_df_y['P']     # Investment per capita
table_two_df_y['r'] = table_two_df_y['FR'] - table_two_df_y['CPI']    # Real interest rate
table_two_df_y['Y/N'] = table_two_df_y['Y'] - table_two_df_y['N']
table_two_df_y = table_two_df_y.drop(columns=['IT', 'P', 'CPI', 'FR'])
table_two_df_y.to_csv('table_two_y_final.csv')


for column in table_one_df.columns:
    table_one_df[column] = np.log(table_one_df[column])

for column in table_two_df_y.columns:
    table_two_df_y[column] = np.log(table_two_df_y[column])

for column in table_one_df.columns:
    cycle, _ = hpfilter(table_one_df[column], lamb=1600)
    table_one_df[column] = cycle

for column in table_two_df_y.columns:
    cycle, _ = hpfilter(table_two_df_y[column], lamb=1600)
    table_two_df_y[column] = cycle


volatility_df = pd.DataFrame({
    'GDP': table_one_df['GDP'],
    'CND': table_one_df['CND'],
    'CD': table_one_df['CD'],
    'H': table_one_df['H'],
    'AveH': table_one_df['AveH'],
    'L': table_one_df['L'],
    'AveW': table_one_df['AveW'],
    'GDP/L': table_one_df['GDP/L'],
    'C': table_two_df_q['C'],
    'I': table_two_df_q['I'],
    'A': table_two_df_y['A'],
    'r': table_two_df_q['r'],
    'Y/N': table_two_df_q['Y/N']
})

volatility = volatility_df.std()
volatility_df = pd.DataFrame(volatility, columns=['Volatility'])
latex_code = volatility_df.to_latex()
print(latex_code)
with open('volatility_table.tex', 'w') as file:
    file.write(latex_code)