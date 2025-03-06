import time
from collections import defaultdict
import pandas as pd
import numpy as np

# %%

data = pd.read_csv('../data/raw_customer_visists_filtered.csv',
                   parse_dates=['date'])

# %%

start_time = time.time()

# %%

daily = (data
         .groupby([
             'venue_chain_id',
             'venue_id',
             'user_id',
             pd.Grouper(key='date', freq='D')])
         .size()
         .rename('num_visits')
         .reset_index()
         .assign(num_visits=1))

daily['diff'] = (daily
                 .groupby([
                     'venue_chain_id',
                     'venue_id',
                     'user_id'])['date']
                 .diff().dt.days)

daily['is_non_return'] = np.where(
    (daily['user_id'] != 'CASH') & (daily['diff'] > 365), 1, 0)

# %%

monthly = (daily
           .groupby([
               'venue_chain_id',
               'venue_id',
               pd.Grouper(key='date', freq='ME'),
               'user_id'],
               sort=False)['num_visits']
           .sum()
           .reset_index())

# %%

monthly_counts = (monthly
                  .groupby([
                      'venue_chain_id',
                      'venue_id',
                      'date'])
                  .apply(lambda x:
                         dict(zip(x['user_id'], x['num_visits'])),
                         include_groups=False)
                  .rename('user_counts')
                  .to_frame())


# %%

def expand_counts(exp_series):
    res = defaultdict(int)
    for d in exp_series:
        for k, v in d.items():
            res[k] += v
    return dict(res)


monthly_counts['exp_counts'] = (
    monthly_counts
    .groupby([
        'venue_chain_id',
        'venue_id'
    ])['user_counts']
    .transform(lambda g:
               (g.apply(lambda x: [x])
                .cumsum()
                .apply(expand_counts))))


# %%

monthly_non_return = (daily
                      .groupby([
                          'venue_chain_id',
                          'venue_id',
                          pd.Grouper(key='date', freq='ME')],
                          sort=False)['user_id']
                      .apply(set)
                      .rename('non_return_set'))

monthly_counts = monthly_counts.join(monthly_non_return)

# to be further optimised
# logic needs to be checked
# possibly should make adjustnebt in expand_counts
monthly_counts['exp_counts'] = (
    monthly_counts
    .apply(lambda r:
           {k: v for k, v in r['exp_counts'].items()
            if k not in r['non_return_set']},
           axis=1))

# %%

monthly_counts['exp_counts'] = monthly_counts['exp_counts'].shift()

monthly_counts.dropna(inplace=True)

# %%

demo = pd.DataFrame({
    'date': pd.to_datetime([
        '2024-01-31',
        '2024-02-29',
        '2024-03-31',
        '2024-04-30']),
    'user_counts': [
        {'A': 1, 'B': 1},
        {'B': 3, 'C': 4},
        {'A': 2, 'C': 1, 'D': 5},
        {'B': 2, 'D': 3, 'E': 4}
    ]
})

demo['exp_counts'] = (demo['user_counts']
                      .apply(lambda x: [x])
                      .cumsum()
                      .apply(expand_counts)
                      .shift())

demo.dropna(inplace=True)

demo[['new', 'return', 'repeat']] = (
    demo
    .apply(
        lambda r:
            pd.Series({
                'new': set(r['user_counts']) - set(r['exp_counts']),
                'return': set(r['user_counts']) & {k for k, v in r['exp_counts'].items() if v == 1},
                'repeat': set(r['user_counts']) & {k for k, v in r['exp_counts'].items() if v >= 2}
            }),
        axis=1
    ))

# %%

stats = (
    monthly_counts
    .apply(
        lambda r:
            pd.Series({
                'new': len(set(r['user_counts']) - set(r['exp_counts'])) + r['user_counts']['CASH'],
                'return': len(set(r['user_counts']) & {k for k, v in r['exp_counts'].items() if v == 1}),
                'return_id': set(r['user_counts']) & {k for k, v in r['exp_counts'].items() if v == 1},
                'repeat': len(set(r['user_counts']) & {k for k, v in r['exp_counts'].items() if v >= 2})
            }),
        axis=1
    ))

# %%

conversion = (stats
              .explode('return_id')
              .reset_index()
              .drop(['new', 'return', 'repeat'],
                    axis=1)
              .pipe(lambda x:
                    x[x['return_id'] != 'CASH']))

conversion = (conversion
              .merge(
                  (daily[[
                      'venue_chain_id',
                      'venue_id',
                      'user_id',
                      'diff']]
                   .dropna()
                   .drop_duplicates(['venue_chain_id',
                                     'venue_id',
                                     'user_id'
                                     ])
                   ),
                  left_on=['venue_chain_id', 'venue_id', 'return_id'],
                  right_on=['venue_chain_id', 'venue_id', 'user_id']))

conversion['diff_bins'] = pd.cut(
    conversion['diff'], [-np.inf, 30, 60, 90, 180, 365, np.inf])

result = (conversion
          .groupby([
              'venue_chain_id',
              'venue_id',
              'date', 'diff_bins'],
              observed=False)
          .size()
          .unstack(fill_value=0))

# %%

stats['return_repeat'] = stats['return'] + stats['repeat']



stats[['new_12m_mean', 'return_12m_mean']] = (
    stats
    .groupby([
        'venue_chain_id',
        'venue_id'])[['new', 'return_repeat']]
    .apply(lambda g:
           g.rolling(
               window=12,
               min_periods=12)
           .mean())
    .reset_index(level=[0, 1], drop=True)
)

stats_out = (stats[[
    'new',
    'return',
    'repeat',
    'new_12m_mean',
    'return_12m_mean'
]]
    .dropna())

# %%

end_time = time.time()
print(f'Elapsed time: {end_time - start_time:.2f} sec')

# %%

stats_out.to_csv('../derived/stats.csv')
result.to_csv('../derived/conversion.csv')
