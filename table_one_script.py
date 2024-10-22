import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import dask.dataframe as dd
import sys

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

# Filter records where avg_posts > 600
print("Filtering records where avg_posts > 600...")
filtered_data = data[data['avg_posts'] > 600]
print("Records filtered by avg_posts.")

# Keep users who had total views > 200 in the pre period (Dec)
print("Filtering records where total_views_pre > 200...")
filtered_data = filtered_data[filtered_data['total_views_pre'] > 200]
print("Records filtered by total_views_pre.")

# Have to compute the filtered data to convert it to a pandas DataFrame to use with smf
print("Computing filtered data to convert to pandas DataFrame...")
filtered_data = filtered_data.compute()
print("Filtered data computed.")

D1 = pd.get_dummies(filtered_data["city"], prefix="city", drop_first=True)
D2 = pd.get_dummies(filtered_data["state"], prefix="state", drop_first=True)
D3 = pd.get_dummies(filtered_data["gender"], prefix="gender", drop_first=True)



# Confirm all columns are of type float32
print("Confirming all columns are of type float32...")
for col in filtered_data.select_dtypes(include=['float']).columns:
    filtered_data[col] = filtered_data[col].astype('float32')
print("All columns confirmed as float32.")

print("Running...")
# bucket_formula = 'bucketFeed ~ D1 + D2 + D3
# trending_formula = 'trendingFeed ~  D1 + D2 + D3
X = pd.concat([D1, D2, D3], axis=1).fillna(0).astype(float)  # Concatenate the dummy variables

# D3 = D3.astype(int)  # Converts boolean to 0s and 1s if D1 is boolean
filtered_data['trendingFeed'] = filtered_data['trendingFeed'].fillna(0).astype(float)  # Same for bucketFeed

# X = np.array(X)
# .reshape(-1, 1)  # Reshape D1 into a 2D array
X = sm.add_constant(X)  # Add a constant to the reshaped array
# y = np.array(filtered_data['bucketFeed'])  # Keep y as a 1D array
y = filtered_data['trendingFeed']

# print("Are there any NaN values in X?", np.isnan(X).any())
# print("Are there any NaN values in y?", np.isnan(y).any())

# print("Shape of X: ", X.shape)
# print("Shape of y: ", y.shape)

result = sm.OLS(y, X).fit()
print(result.summary())

# Write the summary to a text file
with open('tables/table_one.txt', 'w') as f:
    f.write(result.summary().as_text())

coefficients = result.params
std_errors = result.bse
p_values = result.pvalues

# Create a DataFrame to hold the OLS results
ols_results = pd.DataFrame({
    'Coefficient': coefficients,
    'Standard Error': std_errors,
    'p-value': p_values
})

# Filter the coefficients within the specified ranges
filtered_results = ols_results[((ols_results['Coefficient'] > 0.0001) & (ols_results['Coefficient'] < 1)) |
                               ((ols_results['Coefficient'] > -1) & (ols_results['Coefficient'] < -0.0001))]
print(filtered_results)

sys.exit()

# Regression for bucketFeed (apparently -1 in the formula leads to no intercept)
print("Running regression for bucketFeed...")
bucketFeed_model = smf.ols(bucket_formula, data=filtered_data).fit()
print("bucketFeed_model.summary(): ", bucketFeed_model.summary())



bucket_f_test_result = bucketFeed_model.f_test(np.identity(len(bucketFeed_model.params)))
print("bucketFeed f_test_result: ", bucket_f_test_result)

# Regression for trendingFeed
print("Running regression for trendingFeed...")
trendingFeed_model = smf.ols(trending_formula, data=filtered_data).fit()
print("trendingFeed_model.summary(): ", trendingFeed_model.summary())

trending_f_test_result = trendingFeed_model.f_test(np.identity(len(trendingFeed_model.params)))
print("trendingFeed f_test_result: ", trending_f_test_result)

# np.random.seed(42)
# selected_characteristics = np.random.choice(independent_vars_list, size=5, replace=False)
# print("Selected characteristics:", selected_characteristics)

# # Extract the indices of the selected characteristics
# selected_indices = [independent_vars_list.index(char) for char in selected_characteristics]

# # Get the corresponding results from f_test_result
# # bucketFeed_f_test_selected = bucketFeed_model.effects[selected_indices]
# # trendingFeed_f_test_selected = trendingFeed_model.effects[selected_indices]
# # print("bucketFeed f_test_result for selected characteristics:", bucketFeed_f_test_selected)
# # print("trendingFeed f_test_result for selected characteristics:", trendingFeed_f_test_selected)

# # bucketFeed_coeffs_selected = bucketFeed_model.params[selected_indices]
# # trendingFeed_coeffs_selected = trendingFeed_model.params[selected_indices]
# # Extract coefficients and standard errors for bucketFeed
# bucketFeed_coeffs_selected = bucketFeed_model.params[selected_indices]
# bucketFeed_se_selected = bucketFeed_model.bse[selected_indices]
# bucketFeed_pvalues_selected = bucketFeed_model.pvalues[selected_indices]

# # Extract coefficients and standard errors for trendingFeed
# trendingFeed_coeffs_selected = trendingFeed_model.params[selected_indices]
# trendingFeed_se_selected = trendingFeed_model.bse[selected_indices]
# trendingFeed_pvalues_selected = trendingFeed_model.pvalues[selected_indices]

# # Print the results for bucketFeed
# print("BucketFeed Model:")
# for i, char in enumerate(selected_characteristics):
#     print(f"{char}: Coefficient = {bucketFeed_coeffs_selected[i]:.3f}, SE = {bucketFeed_se_selected[i]:.3f}, p-value = {bucketFeed_pvalues_selected[i]:.3f}")

# # Print the results for trendingFeed
# print("TrendingFeed Model:")
# for i, char in enumerate(selected_characteristics):
#     print(f"{char}: Coefficient = {trendingFeed_coeffs_selected[i]:.3f}, SE = {trendingFeed_se_selected[i]:.3f}, p-value = {trendingFeed_pvalues_selected[i]:.3f}")


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
