import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import dask.dataframe as dd

print("Starting script...")

# Define the correct data types for the columns
dtype = {
    'bucketFeed': 'float32',
    'hln_any': 'float32',
    'num_login': 'float32',
    'total_views': 'float32',
    'trendingFeed': 'float32',
}

# Load the data from CSV using Dask for large datasets (this didn't end up mattering)
data_path = 'data/treatment_outcome_lunch.csv'  
print(f"Loading data from {data_path}...")
data = dd.read_csv(data_path, dtype=dtype)
print("Data loaded.")

# Clean column names and values (there can't be spaces)
print("Cleaning column names and values...")
data.columns = data.columns.str.replace(" ", "_")
data["city"] = data["city"].str.replace(" ", "_")
data["state"] = data["state"].str.replace(" ", "_")
data["gender"] = data["gender"].str.replace(" ", "_")
data["language"] = data["language"].str.replace(" ", "_")
data["ageRange"] = data["ageRange"].str.replace(" ", "_")

# Replace hyphens with underscores since it is interpreted as a minus symbol
data["ageRange"] = data["ageRange"].str.replace("-", "_")

# Replace addition symbol with "add" since "+" is interpreted as addition
data["ageRange"] = data["ageRange"].str.replace("+", "add")
print("Column names and values cleaned.")

# Convert columns to category
print("Converting columns to category...")
data["city"] = data["city"].astype('category')
data["state"] = data["state"].astype('category')
data["gender"] = data["gender"].astype('category')
data["language"] = data["language"].astype('category')
data["ageRange"] = data["ageRange"].astype('category')

data = data.categorize(columns=["city", "state", "gender"])
print("Columns converted to category.")

# Filter records where avg_posts > 600
print("Filtering records where avg_posts > 600...")
filtered_data = data[data['avg_posts'] > 600]
print("Records filtered by avg_posts.")

# Keep users who had total views > 200 in the pre period (Dec)
print("Filtering records where total_views_pre > 200...")
filtered_data = filtered_data[filtered_data['total_views_pre'] > 200]
print("Records filtered by total_views_pre.")

# Create dummy variables for city, state, gender, language, and ageRange
print("Creating dummy variables...")
dummies = dd.get_dummies(filtered_data[['city', 'state', 'gender']], drop_first=True)
filtered_data = dd.concat([filtered_data, dummies], axis=1)
print("Dummy variables created.")

# Setting up the independent variables for the regression
print("Setting up independent variables for regression...")
independent_vars_list = dummies.columns.tolist()
independent_vars = ' + '.join(independent_vars_list)
print("Independent variables set up.")

# Have to compute the filtered data to convert it to a pandas DataFrame to use with smf
print("Computing filtered data to convert to pandas DataFrame...")
filtered_data = filtered_data.compute()
print("Filtered data computed.")

# Confirm all columns are of type float32
print("Confirming all columns are of type float32...")
for col in filtered_data.select_dtypes(include=['float']).columns:
    filtered_data[col] = filtered_data[col].astype('float32')
print("All columns confirmed as float32.")

# Regression for bucketFeed (apparently -1 in the formula leads to no intercept)
print("Running regression for bucketFeed...")
bucketFeed_model = smf.ols(f'bucketFeed ~ {independent_vars}', data=filtered_data).fit()
print("bucketFeed_model.summary(): ", bucketFeed_model.summary())

bucket_f_test_result = bucketFeed_model.f_test(np.identity(len(bucketFeed_model.params)))
print("bucketFeed f_test_result: ", bucket_f_test_result)

# Regression for trendingFeed
print("Running regression for trendingFeed...")
trendingFeed_model = smf.ols(f'trendingFeed ~ {independent_vars}', data=filtered_data).fit()
print("trendingFeed_model.summary(): ", trendingFeed_model.summary())

trending_f_test_result = trendingFeed_model.f_test(np.identity(len(trendingFeed_model.params)))
print("trendingFeed f_test_result: ", trending_f_test_result)

np.random.seed(42)
selected_characteristics = np.random.choice(independent_vars_list, size=5, replace=False)
print("Selected characteristics:", selected_characteristics)

# Extract the indices of the selected characteristics
selected_indices = [independent_vars_list.index(char) for char in selected_characteristics]

# Get the corresponding results from f_test_result
# bucketFeed_f_test_selected = bucketFeed_model.effects[selected_indices]
# trendingFeed_f_test_selected = trendingFeed_model.effects[selected_indices]
# print("bucketFeed f_test_result for selected characteristics:", bucketFeed_f_test_selected)
# print("trendingFeed f_test_result for selected characteristics:", trendingFeed_f_test_selected)

# bucketFeed_coeffs_selected = bucketFeed_model.params[selected_indices]
# trendingFeed_coeffs_selected = trendingFeed_model.params[selected_indices]
# Extract coefficients and standard errors for bucketFeed
bucketFeed_coeffs_selected = bucketFeed_model.params[selected_indices]
bucketFeed_se_selected = bucketFeed_model.bse[selected_indices]
bucketFeed_pvalues_selected = bucketFeed_model.pvalues[selected_indices]

# Extract coefficients and standard errors for trendingFeed
trendingFeed_coeffs_selected = trendingFeed_model.params[selected_indices]
trendingFeed_se_selected = trendingFeed_model.bse[selected_indices]
trendingFeed_pvalues_selected = trendingFeed_model.pvalues[selected_indices]

# Print the results for bucketFeed
print("BucketFeed Model:")
for i, char in enumerate(selected_characteristics):
    print(f"{char}: Coefficient = {bucketFeed_coeffs_selected[i]:.3f}, SE = {bucketFeed_se_selected[i]:.3f}, p-value = {bucketFeed_pvalues_selected[i]:.3f}")

# Print the results for trendingFeed
print("TrendingFeed Model:")
for i, char in enumerate(selected_characteristics):
    print(f"{char}: Coefficient = {trendingFeed_coeffs_selected[i]:.3f}, SE = {trendingFeed_se_selected[i]:.3f}, p-value = {trendingFeed_pvalues_selected[i]:.3f}")


# # Extract coefficients and standard errors
# print("Extracting coefficients and standard errors for bucketFeed...")
# bucketFeed_coeffs = bucketFeed_model.params[selected_characteristics]
# bucketFeed_se = bucketFeed_model.bse[selected_characteristics]

# print("Extracting coefficients and standard errors for trendingFeed...")
# trendingFeed_coeffs = trendingFeed_model.params[selected_characteristics]
# trendingFeed_se = trendingFeed_model.bse[selected_characteristics]

# # Create the balance table
# print("Creating balance table...")
# balance_table = pd.DataFrame({
#     'Characteristic': selected_characteristics,
#     'Coefficient wrt Bucket Feed (SE)': [f'{coef:.3f} ({se:.3f})' for coef, se in zip(bucketFeed_coeffs, bucketFeed_se)],
#     'Coefficient wrt Trending Feed (SE)': [f'{coef:.3f} ({se:.3f})' for coef, se in zip(trendingFeed_coeffs, trendingFeed_se)]
# })

# # Display the balance table
# print("Displaying balance table...")
# print(balance_table)

# print("Script completed.")
