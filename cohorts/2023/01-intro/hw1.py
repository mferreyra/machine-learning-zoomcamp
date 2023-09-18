import pandas as pd
import numpy as np

# Question 1: Version of Pandas
print(pd.__version__)

# Question 2: Number of columns in the dataset
housing_df = pd.read_csv('./housing.csv')
len(housing_df.columns)

# Question 3: Select columns with missing data
housing_df.isna().any()

# Question 4: Number of unique values in the 'ocean_proximity' column
len(housing_df['ocean_proximity'].unique())

# Question 5: Average value of the 'median_house_value' for the houses near the bay
housing_df[housing_df['ocean_proximity'] == 'NEAR BAY']['median_house_value'].mean()

# Question 6: Has the mean value changed after filling missing values?
value_before = housing_df[housing_df['ocean_proximity'] == 'NEAR BAY']['median_house_value'].mean()
value_after = housing_df[housing_df['ocean_proximity'] == 'NEAR BAY'].fillna(housing_df['median_house_value'].median())['median_house_value'].mean()
value_before == value_after

# Questions 7: Value of the last element of w
island_housing = housing_df[housing_df['ocean_proximity'] == 'ISLAND']
X = island_housing[['housing_median_age', 'total_rooms', 'total_bedrooms']].to_numpy()
XTX = X.T @ X
XTX_inverted = np.linalg.inv(XTX)
y = np.array([950, 1300, 800, 1000, 1300])
w = (XTX_inverted @ X.T) @ y
w[-1]
