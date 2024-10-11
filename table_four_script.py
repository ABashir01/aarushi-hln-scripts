import pandas as pd
import statsmodels.formula.api as smf

# Load data as a dataframe
data_path = 'data/treatment_outcome_lunch.csv'
df = pd.read_csv(data_path)

views_model = smf.ols('total_views ~ bucketFeed + trendingFeed', data=df).fit(cov_type='HC0')
logins_model = smf.ols('num_login ~ bucketFeed + trendingFeed', data=df).fit(cov_type='HC0')

result_table = pd.DataFrame({
    "Dependent Variable": ["Bucket Feed Coefficient", "Bucket Feed Standard Error", "Trending Feed Coefficient", "Trending Feed Standard Error"],
    "Total Views, Any Content": [views_model.params["bucketFeed"], f'({views_model.bse["bucketFeed"]})', views_model.params["trendingFeed"], f'({views_model.bse["trendingFeed"]})'],
    "User Retention/Logins": [logins_model.params["bucketFeed"], f'({logins_model.bse["bucketFeed"]})', logins_model.params["trendingFeed"], f'({logins_model.bse["trendingFeed"]})'],
    "Time Spent on Platform": ["N/A", "(N/A)", "N/A", "(N/A)"],
})

print(result_table.to_markdown())

