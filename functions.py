# IMPORTS -----------------------------------------------------------------

# Core libraries for data manipulation and analysis
import pandas as pd  # Data manipulation using DataFrames
import numpy as np  # Numerical and mathematical operations

# Data visualization
import matplotlib.pyplot as plt  # Basic plotting
import seaborn as sns  # Advanced statistical plots
sns.set()  # Set Seaborn's default style

# Statistical and mathematical analysis
from math import ceil  # Round numbers up to the nearest integer
from scipy.stats import f_oneway, skewnorm  # ANOVA test and skew-normal distribution

# Machine learning
from sklearn.cluster import AgglomerativeClustering  # Hierarchical clustering
from sklearn.impute import KNNImputer  # Imputation of missing values using KNN

# Hierarchical clustering and visualization
from scipy.cluster.hierarchy import dendrogram  # Visualize hierarchical clusters

# Other utilities
from os.path import join  # Handle file paths
from itertools import product  # Cartesian product of iterables
from datetime import datetime  # Work with dates
import sqlite3  # SQLite database connection and management
import os  # OS-level operations

# Configuration for high-resolution plots in notebooks
#%config InlineBackend.figure_format = 'retina'





def convert_columns_to_dtype(df, columns, dtype):
    """
    Convert specified columns in a dataframe to a specified dtype.

    Parameters:
    - df (pd.DataFrame): The input dataframe.
    - columns (list): A list of column names to convert.
    - dtype (str): The target dtype for the columns.

    Returns:
    - pd.DataFrame: The dataframe with updated column dtypes.
    """
    for column in columns:
        df[column] = df[column].astype(dtype)



def group_rare_values(df, column, threshold, new_category):
    """
    Group rare values in a specified column into a new category.

    Parameters:
    - df (pd.DataFrame): The input dataframe.
    - column (str): The column to process.
    - threshold (int): The frequency threshold below which values are grouped.
    - new_category (str): The name for the new category (default is 'Other Chains').

    Returns:
    - None: Modifies the dataframe in-place.
    """
    # Calculate the frequency of unique values in the column
    value_counts = df[column].value_counts()

    # Identify rare values that appear less frequently than the threshold
    rare_values = value_counts[value_counts < threshold].index

    # Replace rare values with a new category
    df[column] = df[column].apply(lambda x: new_category if x in rare_values else x)



def plot_category_distribution(df, column, title, figsize, rotation, color):
    """
    Create a bar plot showing the distribution of categories in a specified column.

    Parameters:
    - df (pd.DataFrame): The input dataframe.
    - column (str): The column to visualize.
    - title (str): The title of the plot.
    - figsize (tuple): The size of the figure.
    - rotation (int): The rotation angle for x-axis labels.
    - color (str): The color of the bars.

    Returns:
    - None: Displays the plot.
    """
    # Set the visual style
    plt.style.use('ggplot')
    
    # Create the figure
    plt.figure(figsize=figsize)
    
    # Generate the bar plot
    sns.countplot(
        data=df, 
        x=column, 
        order=df[column].value_counts().index, 
        color=color
    )
    
    # Add title and labels
    plt.title(title)
    plt.xlabel(f'{column} Categories')
    plt.ylabel('Count')
    
    # Rotate x-axis labels
    plt.xticks(rotation=rotation, ha='right')
    
    # Adjust layout to prevent overlapping
    plt.tight_layout()
    
    # Display the plot
    plt.show()



def find_and_count_duplicates_index_version(df):
    """
    Find and count duplicate rows based on the dataframe's index.

    Parameters:
    - df (pd.DataFrame): The input dataframe.

    Returns:
    - None: Prints the count of duplicates for each duplicated index and the total number of duplicated rows.
    """
    # Get the name of the index dynamically
    index_name = df.index.name if df.index.name else "Index"

    # Find duplicates based on the index
    duplicates = df[df.index.duplicated(keep=False)]
    
    # Count occurrences of each duplicated index
    duplicate_counts = duplicates.index.value_counts()
    
    # Count the total number of duplicated rows based on the index
    total_duplicates = duplicate_counts.sum()
    
    print(f"Duplicate count for each '{index_name}' (index of the dataframe):")
    print(duplicate_counts)
    print(f"\nTotal duplicated rows based on '{index_name}': {total_duplicates}")



def remove_duplicates_by_index(df, index_column):
    """
    Remove duplicate rows based on the dataframe's index and verify the cleanup, modifying the dataframe in place.

    Parameters:
    - df (pd.DataFrame): The input dataframe (modified in place).
    - index_column (str): The column to set as the index for duplicate removal.

    Returns:
    - int: The number of remaining duplicates after cleanup.
    """
    # Reset the index to turn the index column into a normal column
    df.reset_index(drop=False, inplace=True)
    
    # Remove duplicate rows
    df.drop_duplicates(inplace=True)
    
    # Set the specified column back as the index
    df.set_index(index_column, inplace=True)
    
    # Verify if duplicates were successfully removed
    remaining_duplicates = df.index.duplicated(keep=False).sum()
    
    print(f"Remaining duplicated rows based on '{index_column}' after cleanup: {remaining_duplicates}")



def calculate_categorical_statistics(df):
    """
    Calculate descriptive statistics for categorical variables in the DataFrame,
    including the "Most Frequent Percentage."

    Parameters:
    - df (pd.DataFrame): The input DataFrame.

    Returns:
    - pd.DataFrame: A DataFrame containing the statistics for each categorical variable.
    """
    # Set display option to show all columns
    pd.set_option('display.max_columns', None)

    # Dictionary to store results
    results = {}

    # Loop through all categorical columns
    for column in df.select_dtypes(include=['object']).columns:
        # Get basic statistics for the column
        stats = df[column].describe()

        # Calculate the percentage of the most frequent value
        top_count = stats['freq']
        total_count = len(df[column])
        most_frequent_percentage = (top_count / total_count) * 100

        # Store statistics and "Most Frequent Percentage" in the results dictionary
        results[column] = {
            'Count': stats['count'],
            'Unique': stats['unique'],
            'Top': stats['top'],
            'Freq': top_count,
            'Most Frequent Percentage': most_frequent_percentage
        }

    # Convert results to DataFrame and return transposed for readability
    final_statistics = pd.DataFrame(results).T
    return final_statistics



def display_value_counts_by_type(df, dtypes):
    """
    Display the value counts for columns of specified data types in the dataframe.

    Parameters:
    - df (pd.DataFrame): The input dataframe.
    - dtypes (tuple): A tuple of data types to filter columns (e.g., ('object',), ('integer', 'float')).

    Returns:
    - None: Prints the value counts for each column of the specified data types.
    """
    # Select the columns with a specific data types
    selected_columns = df.select_dtypes(include=dtypes).columns

    # Display the value counts for each column
    for column in selected_columns:
        print(f"Value counts for column:")
        print(df[column].value_counts(), "\n", "\n")



def replace_column_value(df, column, old_value, new_value):
    """
    Replace occurrences of a specific value in a given column with a new value.

    Parameters:
    - df (pd.DataFrame): The input dataframe.
    - column (str): The column in which to perform the replacement.
    - old_value: The value to be replaced.
    - new_value: The value to replace with.

    Returns:
    - pd.DataFrame: The dataframe with the replaced values.
    """
    df.loc[df[column] == old_value, column] = new_value



def calculate_order_difference(df):
    """
    Calculate the difference in days between first and last orders.

    Parameters:
    - df (pd.DataFrame): The input dataframe.

    Returns:
    - pd.DataFrame: The dataframe with a new column 'dif_order'.
    """
    df.loc[:, 'dif_order'] = df['last_order'] - df['first_order']


def compute_total_cuisine(df):
    """
    Compute the total amount spent across all cuisine types.

    Parameters:
    - df (pd.DataFrame): The input dataframe.

    Returns:
    - pd.Series: The list of 'tot_CUI' values.
    """
    tot_CUI = [col for col in df.columns if col.startswith('CUI_')]
    return tot_CUI

def add_total_cuisine_to_df(df):
    """
    Add a 'tot_CUI' column to the dataframe based on cuisine types.

    Parameters:
    - df (pd.DataFrame): The input dataframe.

    Returns:
    - pd.DataFrame: The modified dataframe with the 'tot_CUI' column.
    """
    df.loc[:, 'tot_CUI'] = df[compute_total_cuisine(df)].sum(axis=1)



def calculate_work_leisure_days(df):
    """
    Calculate total orders on workdays and leisure days.

    Parameters:
    - df (pd.DataFrame): The input dataframe.

    Returns:
    - pd.DataFrame: The dataframe with new columns 'tot_work_days' and 'tot_leisure_days'.
    """
    work_days = [f'DOW_{i}' for i in range(1, 5)]  # Monday (DOW_1) to Thursday (DOW_4)
    leisure_days = ['DOW_0', 'DOW_5', 'DOW_6']     # Sunday (DOW_0), Friday, and Saturday

    df.loc[:, 'tot_work_days'] = df[work_days].sum(axis=1)
    df.loc[:, 'tot_leisure_days'] = df[leisure_days].sum(axis=1)



def compute_total_products_by_week():
    """
    Compute the total products across all days in a week.

    Parameters:
    - None.

    Returns:
    - pd.Series: A series containing the total products for each row by week.
    """
    total_products_by_week = [f'DOW_{i}' for i in range(7)]
    return total_products_by_week

def compute_total_products_by_day():
    """
    Compute the total products across all hours in a day.

    Parameters:
    - None.

    Returns:
    - pd.Series: A series containing the total products for each row by day.
    """
    total_products_by_day = [f'HR_{i}' for i in range(24)]
    return total_products_by_day

def add_total_products_to_df(df):
    """
    Add columns for total products by week and total products by day to the dataframe.

    Parameters:
    - df (pd.DataFrame): The input dataframe.

    Returns:
    - pd.DataFrame: The modified dataframe with new columns.
    """
    df.loc[:, 'total_products_by_week'] = df[compute_total_products_by_week()].sum(axis=1)
    df.loc[:, 'total_products_by_day'] = df[compute_total_products_by_day()].sum(axis=1)



def calculate_time_periods(df):
    """
    Create columns for total orders per time period of the day.

    Parameters:
    - df (pd.DataFrame): The input dataframe.

    Returns:
    - pd.DataFrame: The dataframe with new columns for time periods.
    """
    df.loc[:, 'tot_early_morning'] = df[[f'HR_{i}' for i in range(0, 6)]].sum(axis=1)   # Midnight to 6 AM
    df.loc[:, 'tot_breakfast'] = df[[f'HR_{i}' for i in range(6, 11)]].sum(axis=1)      # 6 AM to 11 AM
    df.loc[:, 'tot_lunch'] = df[[f'HR_{i}' for i in range(11, 15)]].sum(axis=1)         # 11 AM to 3 PM
    df.loc[:, 'tot_afternoon'] = df[[f'HR_{i}' for i in range(15, 18)]].sum(axis=1)     # 3 PM to 6 PM
    df.loc[:, 'tot_dinner'] = df[[f'HR_{i}' for i in range(18, 22)]].sum(axis=1)        # 6 PM to 10 PM
    df.loc[:, 'tot_late_night'] = df[[f'HR_{i}' for i in range(22, 24)]].sum(axis=1)    # 10 PM to Midnight



def calculate_cuisine_groups(df):
    """
    Calculate total spent for each cuisine group (e.g., Western, Oriental, Other).

    Parameters:
    - df (pd.DataFrame): The input dataframe.

    Returns:
    - pd.DataFrame: The dataframe with new columns for each cuisine group.
    """
    df.loc[:, 'tot_western_cuisines'] = df[['CUI_American', 'CUI_Italian', 'CUI_Cafe', 'CUI_Street Food / Snacks']].sum(axis=1)
    df.loc[:, 'tot_oriental_cuisines'] = df[['CUI_Asian', 'CUI_Chinese', 'CUI_Indian', 'CUI_Japanese', 'CUI_Thai', 'CUI_Noodle Dishes']].sum(axis=1)
    df.loc[:, 'tot_other_cuisines'] = df[['CUI_Beverages', 'CUI_Desserts', 'CUI_Healthy', 'CUI_Chicken Dishes', 'CUI_OTHER']].sum(axis=1)



def check_column_match(df, column_to_check, reference_column):
    """
    Check if a specified column matches a reference column 
    and display the count of matches and mismatches.

    Parameters:
    - df (pd.DataFrame): The input dataframe.
    - column_to_check (str): The name of the column to compare with the reference column.
    - reference_column (str): The column to which the specified column will be compared.

    Returns:
    - pd.Series: A boolean series indicating whether each row matches (True) or mismatches (False).
    """
    # Check if the column matches the reference column
    match_result = df[column_to_check] == df[reference_column]
    
    # Display the match results
    print(f"Matches for '{column_to_check}' compared to '{reference_column}':")
    print(match_result.value_counts())
    print("\n")



def check_missing_and_mismatch(df, column_to_check, total_column_1, total_column_2):
    """
    Check for missing values in a column and rows where totals do not match.

    Parameters:
    - df (pd.DataFrame): The input dataframe.
    - column_to_check (str): The column to check for missing values.
    - total_column_1 (str): The first total column to compare.
    - total_column_2 (str): The second total column to compare.

    Returns:
    - int: Number of missing values in the specified column.
    - int: Number of mismatched rows with missing values in the specified column.
    """
    # Step 1: Count missing values in the specified column
    missing_count = df[column_to_check].isna().sum()
    print(f"Number of missing values in {column_to_check}: {missing_count}")
    
    # Step 2: Create a filter for rows where totals do not match
    mismatch_filter = df[total_column_1] != df[total_column_2]
    
    # Step 3: Combine the filters to find rows with mismatches and missing values
    mismatch_and_missing = df[mismatch_filter & df[column_to_check].isna()]
    mismatch_and_missing_count = mismatch_and_missing.shape[0]
    
    # Step 4: Display the result
    print(f"Number of mismatched rows with missing {column_to_check}: {mismatch_and_missing_count}")

    














    

def calculate_period_trend_by_time(df):
    """
    Creates a DataFrame showing the total orders for each cuisine group across different time periods.

    Parameters:
    df (pd.DataFrame): The original DataFrame.

    Returns:
    pd.DataFrame: A DataFrame with cuisine groups as rows and time periods as columns.
    """
    return pd.DataFrame({
        'Early Morning': [
            df['tot_western_cuisines'].dot(df['tot_early_morning'] > 0),   # Sum of Western cuisine orders during early morning
            df['tot_oriental_cuisines'].dot(df['tot_early_morning'] > 0),  # Sum of Oriental cuisine orders during early morning
            df['tot_other_cuisines'].dot(df['tot_early_morning'] > 0)      # Sum of Other cuisine orders during early morning
        ],
        'Breakfast': [
            df['tot_western_cuisines'].dot(df['tot_breakfast'] > 0),       # Sum of Western cuisine orders during breakfast
            df['tot_oriental_cuisines'].dot(df['tot_breakfast'] > 0),      # Sum of Oriental cuisine orders during breakfast
            df['tot_other_cuisines'].dot(df['tot_breakfast'] > 0)          # Sum of Other cuisine orders during breakfast
        ],
        'Lunch': [
            df['tot_western_cuisines'].dot(df['tot_lunch'] > 0),           # Sum of Western cuisine orders during lunch
            df['tot_oriental_cuisines'].dot(df['tot_lunch'] > 0),          # Sum of Oriental cuisine orders during lunch
            df['tot_other_cuisines'].dot(df['tot_lunch'] > 0)              # Sum of Other cuisine orders during lunch
        ],
        'Afternoon': [
            df['tot_western_cuisines'].dot(df['tot_afternoon'] > 0),       # Sum of Western cuisine orders during afternoon
            df['tot_oriental_cuisines'].dot(df['tot_afternoon'] > 0),      # Sum of Oriental cuisine orders during afternoon
            df['tot_other_cuisines'].dot(df['tot_afternoon'] > 0)          # Sum of Other cuisine orders during afternoon
        ],
        'Dinner': [
            df['tot_western_cuisines'].dot(df['tot_dinner'] > 0),          # Sum of Western cuisine orders during dinner
            df['tot_oriental_cuisines'].dot(df['tot_dinner'] > 0),         # Sum of Oriental cuisine orders during dinner
            df['tot_other_cuisines'].dot(df['tot_dinner'] > 0)             # Sum of Other cuisine orders during dinner
        ],
        'Late Night': [
            df['tot_western_cuisines'].dot(df['tot_late_night'] > 0),      # Sum of Western cuisine orders during late night
            df['tot_oriental_cuisines'].dot(df['tot_late_night'] > 0),     # Sum of Oriental cuisine orders during late night
            df['tot_other_cuisines'].dot(df['tot_late_night'] > 0)         # Sum of Other cuisine orders during late night
        ]
    }, index=['Western', 'Oriental', 'Other'])



def calculate_period_trend_by_cuisine(df):
    """
    Creates a DataFrame showing the total orders for each time period across different cuisine groups.

    Parameters:
    df (pd.DataFrame): The original DataFrame.

    Returns:
    pd.DataFrame: A DataFrame with time periods as rows and cuisine groups as columns.
    """
    return pd.DataFrame({
        'Western': [
            df['tot_western_cuisines'].dot(df['tot_early_morning'] > 0),   # Sum of Western cuisine orders during early morning
            df['tot_western_cuisines'].dot(df['tot_breakfast'] > 0),       # Sum of Western cuisine orders during breakfast
            df['tot_western_cuisines'].dot(df['tot_lunch'] > 0),           # Sum of Western cuisine orders during lunch
            df['tot_western_cuisines'].dot(df['tot_afternoon'] > 0),       # Sum of Western cuisine orders during afternoon
            df['tot_western_cuisines'].dot(df['tot_dinner'] > 0),          # Sum of Western cuisine orders during dinner
            df['tot_western_cuisines'].dot(df['tot_late_night'] > 0)       # Sum of Western cuisine orders during late night
        ],
        'Oriental': [
            df['tot_oriental_cuisines'].dot(df['tot_early_morning'] > 0),  # Sum of Oriental cuisine orders during early morning
            df['tot_oriental_cuisines'].dot(df['tot_breakfast'] > 0),      # Sum of Oriental cuisine orders during breakfast
            df['tot_oriental_cuisines'].dot(df['tot_lunch'] > 0),          # Sum of Oriental cuisine orders during lunch
            df['tot_oriental_cuisines'].dot(df['tot_afternoon'] > 0),      # Sum of Oriental cuisine orders during afternoon
            df['tot_oriental_cuisines'].dot(df['tot_dinner'] > 0),         # Sum of Oriental cuisine orders during dinner
            df['tot_oriental_cuisines'].dot(df['tot_late_night'] > 0)      # Sum of Oriental cuisine orders during late night
        ],
        'Other': [
            df['tot_other_cuisines'].dot(df['tot_early_morning'] > 0),     # Sum of Other cuisine orders during early morning
            df['tot_other_cuisines'].dot(df['tot_breakfast'] > 0),         # Sum of Other cuisine orders during breakfast
            df['tot_other_cuisines'].dot(df['tot_lunch'] > 0),             # Sum of Other cuisine orders during lunch
            df['tot_other_cuisines'].dot(df['tot_afternoon'] > 0),         # Sum of Other cuisine orders during afternoon
            df['tot_other_cuisines'].dot(df['tot_dinner'] > 0),            # Sum of Other cuisine orders during dinner
            df['tot_other_cuisines'].dot(df['tot_late_night'] > 0)         # Sum of Other cuisine orders during late night
        ]
    }, index=['Early Morning', 'Breakfast', 'Lunch', 'Afternoon', 'Dinner', 'Late Night'])



def calculate_work_leisure_trend_by_daytype(df):
    """
    Creates a DataFrame showing total orders on work and leisure days by cuisine groups.

    Parameters:
    df (pd.DataFrame): The original DataFrame.

    Returns:
    pd.DataFrame: A DataFrame with cuisine groups as columns and day types as rows.
    """
    return pd.DataFrame({
        'Western': [
            df['tot_western_cuisines'].dot(df['tot_work_days'] > 0),    # Total Western cuisine orders on work days
            df['tot_western_cuisines'].dot(df['tot_leisure_days'] > 0)  # Total Western cuisine orders on leisure days
        ],
        'Oriental': [
            df['tot_oriental_cuisines'].dot(df['tot_work_days'] > 0),   # Total Oriental cuisine orders on work days
            df['tot_oriental_cuisines'].dot(df['tot_leisure_days'] > 0) # Total Oriental cuisine orders on leisure days
        ],
        'Other': [
            df['tot_other_cuisines'].dot(df['tot_work_days'] > 0),      # Total Other cuisine orders on work days
            df['tot_other_cuisines'].dot(df['tot_leisure_days'] > 0)    # Total Other cuisine orders on leisure days
        ]
    }, index=['Work Days', 'Leisure Days'])



def calculate_work_leisure_trend_by_cuisine(df):
    """
    Creates a DataFrame showing total orders on work and leisure days by day types.

    Parameters:
    df (pd.DataFrame): The original DataFrame.

    Returns:
    pd.DataFrame: A DataFrame with day types as columns and cuisine groups as rows.
    """
    return pd.DataFrame({
        'Work Days': [
            df['tot_western_cuisines'].dot(df['tot_work_days'] > 0),     # Total Western cuisine orders on work days
            df['tot_oriental_cuisines'].dot(df['tot_work_days'] > 0),    # Total Oriental cuisine orders on work days
            df['tot_other_cuisines'].dot(df['tot_work_days'] > 0)        # Total Other cuisine orders on work days
        ],
        'Leisure Days': [
            df['tot_western_cuisines'].dot(df['tot_leisure_days'] > 0),  # Total Western cuisine orders on leisure days
            df['tot_oriental_cuisines'].dot(df['tot_leisure_days'] > 0), # Total Oriental cuisine orders on leisure days
            df['tot_other_cuisines'].dot(df['tot_leisure_days'] > 0)     # Total Other cuisine orders on leisure days
        ]
    }, index=['Western', 'Oriental', 'Other'])



def calculate_period_trend_by_day_type(df):
    """
    Creates a DataFrame that calculates the total orders for each time period on work and leisure days,
    using Day Types as columns.

    Parameters:
    df (pd.DataFrame): The original DataFrame.

    Returns:
    pd.DataFrame: A DataFrame with Day Types (Work Days and Leisure Days) as columns and Time Periods as rows.
    """
    return pd.DataFrame({
        'Work Days': [
            df['tot_early_morning'].dot(df['tot_work_days'] > 0),     # Total early morning orders on work days
            df['tot_breakfast'].dot(df['tot_work_days'] > 0),         # Total breakfast orders on work days
            df['tot_lunch'].dot(df['tot_work_days'] > 0),             # Total lunch orders on work days
            df['tot_afternoon'].dot(df['tot_work_days'] > 0),         # Total afternoon orders on work days
            df['tot_dinner'].dot(df['tot_work_days'] > 0),            # Total dinner orders on work days
            df['tot_late_night'].dot(df['tot_work_days'] > 0)         # Total late night orders on work days
        ],
        'Leisure Days': [
            df['tot_early_morning'].dot(df['tot_leisure_days'] > 0),  # Total early morning orders on leisure days
            df['tot_breakfast'].dot(df['tot_leisure_days'] > 0),      # Total breakfast orders on leisure days
            df['tot_lunch'].dot(df['tot_leisure_days'] > 0),          # Total lunch orders on leisure days
            df['tot_afternoon'].dot(df['tot_leisure_days'] > 0),      # Total afternoon orders on leisure days
            df['tot_dinner'].dot(df['tot_leisure_days'] > 0),         # Total dinner orders on leisure days
            df['tot_late_night'].dot(df['tot_leisure_days'] > 0)      # Total late night orders on leisure days
        ]
    }, index=['Early Morning', 'Breakfast', 'Lunch', 'Afternoon', 'Dinner', 'Late Night'])



def calculate_period_trend_by_time_period(df):
    """
    Creates a DataFrame that calculates the total orders for each time period on work and leisure days,
    using Time Periods as columns.

    Parameters:
    df (pd.DataFrame): The original DataFrame.

    Returns:
    pd.DataFrame: A DataFrame with Time Periods (Early Morning, Breakfast, etc.) as columns and Day Types as rows.
    """
    return pd.DataFrame({
        'Early Morning': [
            df['tot_early_morning'].dot(df['tot_work_days'] > 0),    # Total early morning orders on work days
            df['tot_early_morning'].dot(df['tot_leisure_days'] > 0)  # Total early morning orders on leisure days
        ],
        'Breakfast': [
            df['tot_breakfast'].dot(df['tot_work_days'] > 0),        # Total breakfast orders on work days
            df['tot_breakfast'].dot(df['tot_leisure_days'] > 0)      # Total breakfast orders on leisure days
        ],
        'Lunch': [
            df['tot_lunch'].dot(df['tot_work_days'] > 0),            # Total lunch orders on work days
            df['tot_lunch'].dot(df['tot_leisure_days'] > 0)          # Total lunch orders on leisure days
        ],
        'Afternoon': [
            df['tot_afternoon'].dot(df['tot_work_days'] > 0),        # Total afternoon orders on work days
            df['tot_afternoon'].dot(df['tot_leisure_days'] > 0)      # Total afternoon orders on leisure days
        ],
        'Dinner': [
            df['tot_dinner'].dot(df['tot_work_days'] > 0),           # Total dinner orders on work days
            df['tot_dinner'].dot(df['tot_leisure_days'] > 0)         # Total dinner orders on leisure days
        ],
        'Late Night': [
            df['tot_late_night'].dot(df['tot_work_days'] > 0),       # Total late night orders on work days
            df['tot_late_night'].dot(df['tot_leisure_days'] > 0)     # Total late night orders on leisure days
        ]
    }, index=['Work Days', 'Leisure Days'])



def number_bins_sturges(data):
    '''
    Calculates the number of bins based on the number of data points, using Sturges' rule
    Sturges' rule: k = log2(n) + 1

    Requires: The dataset for which the number of bins is to be calculated.
    Ensures:
        - The returned value is a positive integer representing the number of bins.
        - The number of bins increases logarithmically as the dataset size increases.
    '''

    n = len(data)
    bins = np.ceil(np.log2(n) + 1) # np.log2 computes the base-2 logarithm of n, and np.ceil rounds the result up to the next whole number.
    return int(bins)



import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import skewnorm

def plot_histograms_with_mean_median(data, features, title_prefix):
    """
    Creates histograms with mean and median lines for the specified features in the dataset.

    Parameters:
    data (pd.DataFrame): The original DataFrame containing the features to plot.
    features (list): List of feature names to plot histograms for.
    title_prefix (str): A prefix for the title of each histogram.

    Returns:
    None: The function displays the histograms directly and does not return a value.
    """
    for feature in features:
        # Check if the feature exists and has non-empty data
        if feature in data.columns and not data[feature].dropna().empty:
            feature_data = data[feature].dropna()  # Remove NaN values

            # Calculate mean and median
            feature_mean = feature_data.mean()
            feature_median = feature_data.median()

            # Plot histogram
            sns.set_theme(style="white", palette=None)
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.histplot(feature_data, bins=20, ax=ax, color='tab:orange')

            # Add mean and median lines
            ax.axvline(feature_median, color='black', linewidth=4,
                       label='Median: {:.2f}'.format(feature_median))
            ax.axvline(feature_mean, color='blue', linestyle='dashed', linewidth=4,
                       label='Mean: {:.2f}'.format(feature_mean))

            # Customize plot
            ax.legend(handlelength=5)
            ax.set_title(f"{title_prefix}: {feature}", fontsize=15)
            plt.show()






def create_limits_table(q1, q3, iqr, lower_lim, upper_lim):
    """
    Create a DataFrame to display Q1, Q3, IQR, lower limit, and upper limit.

    Parameters:
    - q1 (pd.Series): First quartile values.
    - q3 (pd.Series): Third quartile values.
    - iqr (pd.Series): Interquartile range values.
    - lower_lim (pd.Series): Lower limit values.
    - upper_lim (pd.Series): Upper limit values.

    Returns:
    - pd.DataFrame: Table containing Q1, Q3, IQR, lower limit, and upper limit.
    """
    limits_table = pd.DataFrame({
        'Q1 (25%)': q1,
        'Q3 (75%)': q3,
        'IQR': iqr,
        'Lower Limit': lower_lim,
        'Upper Limit': upper_lim
    })
    return limits_table






















