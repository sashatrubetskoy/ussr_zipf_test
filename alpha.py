import numpy as np
import pandas as pd
import statsmodels.formula.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

# Read data
df = pd.read_csv('ussr.csv')
df.columns = ['rank_89', 'city', 'pop_89', 'pop_17', 'latest_year', 'rank_17', 
    'rank_change', 'ssr', 'current_name', 'record']

# Take log
cols_to_log = ['rank_89', 'rank_17', 'pop_89', 'pop_17']
for c in cols_to_log:
    df['log_'+c] = df[c].apply(np.log)
df['log_pop_change'] = df['log_pop_17'] - df['log_pop_89']

# Get residuals if Soviet Union dissolved in 1989
df['resid_89'] = np.nan
for ssr in df['ssr'].unique().tolist():
    df_ssr = df[df['ssr'] == ssr]
    model = sm.ols(formula='log_pop_89 ~ log_rank_89', data=df_ssr).fit()

    if len(model.resid) > 5:
        df.loc[df['ssr']==ssr, 'resid_89'] = model.resid

# Calculate additional columns
df['log_resid_89'] = df['resid_89'].apply(np.log)
df['logflip_resid_89'] = -(0.3 - df['resid_89']).apply(np.log)
df['exp_resid_89'] = df['resid_89'].apply(np.exp)
df['log_pop_change_adj'] = df['log_pop_change']
for ssr in df['ssr'].unique().tolist():
    df_ssr = df[df['ssr'] == ssr]
    ssr_mean = df_ssr['log_pop_change'].mean()
    df.loc[(df['ssr'] == ssr), 'log_pop_change_adj'] = df['log_pop_change'] - ssr_mean

# Make plot
# df.plot.scatter(x='log_resid_89', y='log_pop_change')
sns.regplot(x='resid_89', y='log_pop_change_adj', data=df)
plt.savefig('test2.png')
plt.close()