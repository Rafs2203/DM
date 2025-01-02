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

from scipy.stats import chi2_contingency
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA

############################################################################################################
#----------------------------------- Functions used in EDA -----------------------------------------------
############################################################################################################

# =============================
# Section 4
# =============================


# ----- 4.1.2 ----- 

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
                


# ----- 4.2.1 -----
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
    

# ----- 4.2.2 -----
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
    
    

# ----- 4.4 -----
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


# ----- 4.5.1 and 4.5.2 -----
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
    - None: Modifies the dataframe in-place.
    """
    df.loc[df[column] == old_value, column] = new_value
    

# ----- 6.1 -----
# Helper Functions (Computations Only)
def compute_total_products_by_week():
    """
    Generate a list of columns representing the total products across all days in a week.

    Parameters:
    - None.

    Returns:
    - list: A list of column names corresponding to days of the week.
    """
    return [f'DOW_{i}' for i in range(7)]



def compute_total_products_by_day():
    """
    Generate a list of columns representing the total products across all hours in a day.

    Parameters:
    - None.

    Returns:
    - list: A list of column names corresponding to hours of the day.
    """
    return [f'HR_{i}' for i in range(24)]



def compute_total_cuisine(df):
    """
    Generate a list of columns representing total cuisine types.

    Parameters:
    - df (pd.DataFrame): The input dataframe.

    Returns:
    - list: A list of column names that start with 'CUI_'.
    """
    return [col for col in df.columns if col.startswith('CUI_')]



def compute_purchased_cuisines(df):
    """
    Compute the number of cuisines with purchases (values > 0) for each row.

    Parameters:
    - df (pd.DataFrame): The input dataframe.

    Returns:
    - pd.Series: A series containing the count of cuisines with purchases for each row.
    """
    cols_cui = compute_total_cuisine(df)
    return df[cols_cui].apply(lambda row: sum(row > 0), axis=1)



# Main Functions (Adding Columns to the DataFrame)
def add_order_difference_to_df(df):
    """
    Add a 'dif_order' column to the dataframe, representing the difference in days between first and last orders.

    Parameters:
    - df (pd.DataFrame): The input dataframe.

    Returns:
    - None: Modifies the dataframe in-place.
    """
    df.loc[:, 'dif_order'] = df['last_order'] - df['first_order']



def add_total_cuisine_to_df(df):
    """
    Add a 'tot_CUI' column to the dataframe, representing the total amount spent across all cuisine types.

    Parameters:
    - df (pd.DataFrame): The input dataframe.

    Returns:
    - None: Modifies the dataframe in-place.
    """
    df.loc[:, 'tot_CUI'] = df[compute_total_cuisine(df)].sum(axis=1)



def add_work_leisure_days_to_df(df):
    """
    Add 'tot_work_days' and 'tot_leisure_days' columns to the dataframe, representing total orders on workdays and leisure days.

    Parameters:
    - df (pd.DataFrame): The input dataframe.

    Returns:
    - None: Modifies the dataframe in-place.
    """
    work_days = [f'DOW_{i}' for i in range(1, 5)]  # Monday (DOW_1) to Thursday (DOW_4)
    leisure_days = ['DOW_0', 'DOW_5', 'DOW_6']     # Sunday (DOW_0), Friday, and Saturday

    df.loc[:, 'tot_work_days'] = df[work_days].sum(axis=1)
    df.loc[:, 'tot_leisure_days'] = df[leisure_days].sum(axis=1)



def add_total_products_to_df(df):
    """
    Add 'total_products_by_week' and 'total_products_by_day' columns to the dataframe, representing total products ordered per week and per day.

    Parameters:
    - df (pd.DataFrame): The input dataframe.

    Returns:
    - None: Modifies the dataframe in-place.
    """
    df.loc[:, 'total_products_by_week'] = df[compute_total_products_by_week()].sum(axis=1)
    df.loc[:, 'total_products_by_day'] = df[compute_total_products_by_day()].sum(axis=1)



def add_time_periods_to_df(df):
    """
    Add columns to the dataframe representing total orders for specific time periods of the day.

    Columns added:
    - 'tot_early_morning': Midnight to 6 AM.
    - 'tot_breakfast': 6 AM to 11 AM.
    - 'tot_lunch': 11 AM to 3 PM.
    - 'tot_afternoon': 3 PM to 6 PM.
    - 'tot_dinner': 6 PM to 10 PM.
    - 'tot_late_night': 10 PM to Midnight.

    Parameters:
    - df (pd.DataFrame): The input dataframe.

    Returns:
    - None: Modifies the dataframe in-place.
    """
    df.loc[:, 'tot_early_morning'] = df[[f'HR_{i}' for i in range(0, 6)]].sum(axis=1)   # Midnight to 6 AM
    df.loc[:, 'tot_breakfast'] = df[[f'HR_{i}' for i in range(6, 11)]].sum(axis=1)      # 6 AM to 11 AM
    df.loc[:, 'tot_lunch'] = df[[f'HR_{i}' for i in range(11, 15)]].sum(axis=1)         # 11 AM to 3 PM
    df.loc[:, 'tot_afternoon'] = df[[f'HR_{i}' for i in range(15, 18)]].sum(axis=1)     # 3 PM to 6 PM
    df.loc[:, 'tot_dinner'] = df[[f'HR_{i}' for i in range(18, 22)]].sum(axis=1)        # 6 PM to 10 PM
    df.loc[:, 'tot_late_night'] = df[[f'HR_{i}' for i in range(22, 24)]].sum(axis=1)    # 10 PM to Midnight



def add_cuisine_groups_to_df(df):
    """
    Add columns to the dataframe representing total spent for each cuisine group (Western, Oriental, and Other).

    Columns added:
    - 'tot_western_cuisines': Total spent on Western cuisines (e.g., American, Italian).
    - 'tot_oriental_cuisines': Total spent on Oriental cuisines (e.g., Asian, Chinese).
    - 'tot_other_cuisines': Total spent on other cuisines (e.g., Desserts, Beverages).

    Parameters:
    - df (pd.DataFrame): The input dataframe.

    Returns:
    - None: Modifies the dataframe in-place.
    """
    df.loc[:, 'tot_western_cuisines'] = df[['CUI_American', 'CUI_Italian', 'CUI_Cafe', 'CUI_Street Food / Snacks']].sum(axis=1)
    df.loc[:, 'tot_oriental_cuisines'] = df[['CUI_Asian', 'CUI_Chinese', 'CUI_Indian', 'CUI_Japanese', 'CUI_Thai', 'CUI_Noodle Dishes']].sum(axis=1)
    df.loc[:, 'tot_other_cuisines'] = df[['CUI_Beverages', 'CUI_Desserts', 'CUI_Healthy', 'CUI_Chicken Dishes', 'CUI_OTHER']].sum(axis=1)



def add_purchased_cuisines_to_df(df):
    """
    Add a 'purchased_cuisines' column to the dataframe, representing the number of cuisines with purchases.

    Parameters:
    - df (pd.DataFrame): The input dataframe.

    Returns:
    - None: Modifies the dataframe in-place.
    """
    df.loc[:, 'purchased_cuisines'] = compute_purchased_cuisines(df)




# ----- 5.2.1 -----
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


# ----- 5.2.1.1 -----
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
    # Count missing values in the specified column
    missing_count = df[column_to_check].isna().sum()
    print(f"Number of missing values in {column_to_check}: {missing_count}")
    
    # Create a filter for rows where totals do not match
    mismatch_filter = df[total_column_1] != df[total_column_2]
    
    # Combine the filters to find rows with mismatches and missing values
    mismatch_and_missing = df[mismatch_filter & df[column_to_check].isna()]
    mismatch_and_missing_count = mismatch_and_missing.shape[0]
    
    # Display the result
    print(f"Number of mismatched rows with missing {column_to_check}: {mismatch_and_missing_count}")




def fill_missing_values(df, column_name, method):                    # In addition to its use in this section, we have also included it in section 7.2.1.
    """
    Fill missing values in a specified column using a specified method.

    Parameters:
    - df (pd.DataFrame): The input dataframe.
    - column_name (str): The name of the column to fill missing values.
    - method (str or callable): The method to fill missing values:
        - 'median': Fill with the column's median.
        - Callable: A function or lambda that takes the DataFrame and returns a fill value.

    Returns:
    - None: Modifies the dataframe in-place.
    """
    if method == 'median':
        # Fill missing values with the median
        df[column_name] = df[column_name].fillna(df[column_name].median())
    elif callable(method):
        # Fill missing values using a custom callable method
        df[column_name] = df[column_name].fillna(method(df))
    else:
        # Raise an error if the method is not recognized
        raise ValueError(f"Invalid method provided for filling missing values: {method}. "
                         "It must be either 'median' or a callable.")




# ----- 5.2.2 -----
def plot_correlation_heatmaps(df, individual_columns, grouped_columns, individual_title, grouped_title, figsize):
    """
    Plot correlation heatmaps for individual columns and grouped columns.

    Parameters:
    - df (pd.DataFrame): The input dataframe.
    - individual_columns (list or callable): A list of column names or a callable that returns the list of columns for individual correlations.
    - grouped_columns (list): A list of column names for grouped correlations.
    - individual_title (str): Title for the heatmap of individual columns.
    - grouped_title (str): Title for the heatmap of grouped columns.
    - figsize (tuple): Figure size for the subplots.

    Returns:
    - None: Displays the plots.
    """
    # Calculate the correlation for individual columns
    individual_corr = df[individual_columns].corr('spearman') if isinstance(individual_columns, list) else df[individual_columns()].corr('spearman')

    # Calculate the correlation for grouped columns
    grouped_corr = df[grouped_columns].corr('spearman')

    # Set up the figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Plot the heatmap for individual correlations
    sns.heatmap(individual_corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, ax=axes[0])
    axes[0].set_title(individual_title)

    # Plot the heatmap for grouped correlations
    sns.heatmap(grouped_corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, ax=axes[1])
    axes[1].set_title(grouped_title)

    # Adjust layout for better visualization
    plt.tight_layout()
    plt.show()



# ----- 5.2.3.1 -----
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



def plot_side_by_side_barcharts(df1, df2, title1, title2, xlabel1, xlabel2, ylabel1, ylabel2, legend_title1, legend_title2, figsize, color_palette):
    """
    Plot side-by-side bar charts for two DataFrames.

    Parameters:
    - df1 (pd.DataFrame): First DataFrame to plot.
    - df2 (pd.DataFrame): Second DataFrame to plot.
    - title1 (str): Title for the first plot.
    - title2 (str): Title for the second plot.
    - xlabel1 (str): X-axis label for the first plot.
    - xlabel2 (str): X-axis label for the second plot.
    - ylabel1 (str): Y-axis label for the first plot.
    - ylabel2 (str): Y-axis label for the second plot.
    - legend_title1 (str): Legend title for the first plot.
    - legend_title2 (str): Legend title for the second plot.
    - figsize (tuple): Figure size for the plots.
    - color_palette (str): Color palette to use for the bar charts.

    Returns:
    - None: Displays the side-by-side bar charts.
    """
    # Set color palette
    colors = sns.color_palette(color_palette)
    
    # Create side-by-side subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot the first DataFrame
    df1.plot(kind="bar", ax=axes[0], color=colors)
    axes[0].set_title(title1)
    axes[0].set_xlabel(xlabel1)
    axes[0].set_ylabel(ylabel1)
    axes[0].legend(title=legend_title1)
    
    # Plot the second DataFrame
    df2.plot(kind="bar", ax=axes[1], color=colors)
    axes[1].set_title(title2)
    axes[1].set_xlabel(xlabel2)
    axes[1].set_ylabel(ylabel2)
    axes[1].legend(title=legend_title2)
    
    # Adjust layout for better visualization
    plt.tight_layout()
    plt.show()



# ----- 5.2.3.2 -----
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



# ----- 5.2.3.3 -----

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
    

# =============================
# Section 6
# =============================

# ----- 6.1.1  -----

def metric_features_histogram (df, features_groups, title, color, use_log):
    """
    Plot dynamic histograms for specified groups of features with adjustable layout and style.

    Parameters:
    - df (pd.DataFrame): The input dataframe.
    - features_groups (list of lists): A list containing lists of features to plot histograms for.
    - title (str): The main title for the figure. Default is "Numeric Variables' Histograms".
    - color (str): The color for the histograms. Default is "#66c2a5".
    - use_log (bool): Whether to use a log scale for the histograms. Default is False.

    Returns:
    - None: Displays the histograms.
    """
    sns.set()  # Set the Seaborn styling for consistent and clean plots

    # Iterate through each list of feature groups
    for group in features_groups:
        n_rows = ceil(len(group) / 4)  # Calculate the number of rows based on the number of features
        fig, axes = plt.subplots(n_rows, 4, figsize=(15, n_rows * 4), tight_layout=True)  # Prepare subplots

        # Flatten axes and handle cases where axes are fewer than 4 * n_rows
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

        # Plot histograms for each feature in the group
        for ax, feat in zip(axes, group):
            bins_number = number_bins_sturges(df[feat])  # Use 'sturges' method for dynamic binning
            ax.hist(df[feat], bins=bins_number, log=use_log, color=color)
            ax.set_title(feat, y=1.05)  # Set title slightly above the plot

        # Turn off empty subplots
        for ax in axes[len(group):]:
            ax.axis('off')

        # Add a centralized title to the figure
        plt.suptitle(title + (" (Log Scale)" if use_log else ""), fontsize=16)

        # Adjust layout for better visual appearance
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()


# ----- 6.1.2 and 7.3.2  -----
def plot_boxplots(data, title, metric_features):
  '''
  Plots box plots for a set of numeric features in a dataset, organized across multiple figures
  if the number of features exceeds a specified limit. Each figure displays up to 24 box plots.

  Requires:
  - `data` (pd.DataFrame): DataFrame containing numeric features.
    Each feature column should be numeric and may contain NaN values (these will be ignored in plots).
  - `title` (str): Title for the entire figure, displayed at the top of each figure.

  Ensures:
  - Generates and displays one or more figures with box plots, each containing up to 24 subplots (4x6 grid layout).
  - Each figure will have a centralized title and organized layout, with subplots showing the box plot for
    each specified feature in `data`.
  '''
  # Loop through metric features in batches of 24 (one batch per figure)
  for i in range(0, len(metric_features), 24):
      current_features = metric_features[i:i + 24]

      # Create figure and axis grid based on number of features to plot
      fig, axes = plt.subplots(ceil(len(current_features) / 4), 4, figsize=(15, ceil(len(current_features) / 4) * 5))

      # Create a box plot for each feature in the current batch
      for ax, feat in zip(axes.flatten(), current_features):
        if data[feat].dropna().empty:  # Check for empty data
          ax.set_visible(False)      # Hide axes with no data
        else:
          sns.boxplot(x=data[feat], ax=ax, color='#66c2a5')


      plt.suptitle(title, fontsize=20, y=0.95)
      plt.tight_layout(rect=[0, 0, 1, 0.95])
      plt.show()
      
      
# ----- 6.1.3  -----

def plot_values(df, columns, title, xlabel, ylabel, color, figsize, rotation, ha, use_mean):
    """
    Plot the values of specified columns as a bar chart.

    Parameters:
    - df (pd.DataFrame): The input dataframe.
    - columns (list): A list of column names to calculate and plot values for.
    - title (str): Title of the plot.
    - xlabel (str): Label for the x-axis.
    - ylabel (str): Label for the y-axis.
    - color (str): Color for the bars.
    - figsize (tuple): Size of the figure (width, height).
    - rotation (int): Rotation angle for x-axis labels.
    - ha (str): Horizontal alignment for x-axis labels (e.g., 'right', 'center').
    - use_mean (bool): If True, calculates and plots the mean values; otherwise, plots the sum of absolute values.

    Returns:
    - None: Displays the bar chart.
    """
    
    # Calculate values based on the `use_mean` parameter
    if use_mean:
        values = df[columns].mean()  # Calculate the mean values
    else:
        values = df[columns].sum()  # Calculate the sum of absolute values
    
    # Plot the calculated values as a bar chart
    plt.figure(figsize=figsize)
    values.plot(kind='bar', color=color)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=rotation, ha=ha)
    plt.show()
    
# ----- 6.2.1  -----    
def plot_pairwise_relationships(df, features, title, diag_kind, fontsize):
    """
    Plot pairwise relationships between numerical features in a dataframe.

    Parameters:
    - df (pd.DataFrame): The input dataframe.
    - features (list): A list of column names representing numerical features to include in the pairplot.
    - title (str): The title for the pairplot.
    - diag_kind (str): The type of plot to use on the diagonal ('hist' or 'kde').
    - fontsize (int): Font size for the plot title.

    Returns:
    - None: Displays the pairplot.
    """
    # Generate the pairplot
    pairplot = sns.pairplot(df[features], diag_kind=diag_kind)

    # Adjust layout and add a title
    plt.subplots_adjust(top=0.95)
    plt.suptitle(title, fontsize=fontsize)

    # Show the plot
    plt.show()
    
# ----- 6.2.2  -----

def plot_filtered_correlation_heatmap(df, features, method, filter_expr, title, figsize, annot, fmt, cmap):
    # In addition to its use in this section, we have also included it in section 7.4.1
    """
    Plot a filtered heatmap for the correlation matrix of selected features.

    Parameters:
    - df (pd.DataFrame): The input dataframe.
    - features (list): A list of column names to calculate the correlation matrix for.
    - method (str): The correlation method, either 'pearson' (default) or 'spearman'.
    - filter_expr (callable, optional): A function to filter the correlation matrix (e.g., lambda x: abs(x) > 0.6).
    - title (str): The title for the heatmap.
    - figsize (tuple): The size of the figure (width, height).
    - annot (bool): Whether to annotate the heatmap with correlation values.
    - fmt (str): The format for annotation values (e.g., ".2f" for 2 decimal places).
    - cmap (str): The colormap for the heatmap.

    Returns:
    - None: Displays the filtered heatmap.
    """
    # Calculate the correlation matrix
    correlation_matrix = df[features].corr(method=method)
    
    # Apply the filter expression if provided
    if filter_expr:
        correlation_matrix = correlation_matrix.where(filter_expr(correlation_matrix))

    # Set up the figure size for the heatmap
    plt.figure(figsize=figsize)

    # Create the heatmap
    heatmap = sns.heatmap(
        correlation_matrix,
        annot=annot,             # Annotate the heatmap with correlation values
        fmt=fmt,                 # Format numbers to 2 decimal places
        cmap=cmap,               # Use the specified colormap
        cbar_kws={"shrink": .8}  # Adjust the color bar size
    )

    # Set the title for the heatmap
    plt.title(title, fontsize=16)

    # Display the heatmap
    plt.show()
    
# ----- 6.3  -----

def plot_categorical_frequencies(df, features, title, color_or_palette, figsize, rotation):
    """
    Plot bar plots of absolute frequencies for categorical variables.

    Parameters:
    - df (pd.DataFrame): The input dataframe.
    - features (list): A list of categorical features to plot.
    - title (str): Title of the plot.
    - color_or_palette (str or list): Bar color or color palette for the plots. 
      Can be a single color (e.g., 'skyblue') or a list of colors (e.g., sns.color_palette("Set2")).
    - figsize (tuple): Figure size (width, height).
    - rotation (int): Rotation angle for x-axis labels.

    Returns:
    - None: Displays the bar plots.
    """

    sns.set()  # Apply Seaborn styles for clean visuals

    # Determine the layout of subplots
    n_rows = 2  # Fixed number of rows
    n_cols = ceil(len(features) / n_rows)  # Calculate the number of columns dynamically

    # Prepare figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    # Flatten axes for consistent iteration
    axes = axes.flatten()

    # Plot data for each feature
    for idx, (ax, feat) in enumerate(zip(axes, features)):
        # Use a single color or a color from the palette
        if isinstance(color_or_palette, list):
            current_color = color_or_palette[idx % len(color_or_palette)]
        else:
            current_color = color_or_palette

        sns.countplot(x=df[feat], ax=ax, color=current_color)
        ax.set_title(feat)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation)

    # Add a centralized title for the figure
    plt.suptitle(title, fontsize=16)

    # Adjust layout for better appearance
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Display the figure
    plt.show()
    
# ----- 6.4  -----

def plot_categorical_relationships(df, cat1, cat2, colors, figsize):
    """
    Plot the relationship between two categorical variables as stacked bar charts (absolute and relative counts).

    Parameters:
    - df (pd.DataFrame): The input dataframe containing the categorical variables.
    - cat1 (str): The first categorical variable (e.g., 'customer_region').
    - cat2 (str): The second categorical variable (e.g., 'payment_method').
    - colors (list): A list of colors for the bars (e.g., Seaborn color palettes like sns.color_palette("Set2")).
    - figsize (tuple): Size of the figure (width, height). Default is (12, 4).

    Returns:
    - None: Displays the plots.
    """

    sns.set(style="whitegrid")

    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Absolute counts
    abs_counts = df.groupby([cat1, cat2])[cat2].size().unstack()  # Group and count occurrences
    abs_counts.plot.bar(stacked=True, ax=axes[0], color=colors)  # Plot stacked bar chart
    axes[0].set_title(f"{cat1.capitalize()} vs {cat2.capitalize()}, Absolute Counts")
    axes[0].tick_params(axis="x", rotation=45)
    for label in axes[0].get_xticklabels():
        label.set_ha("right")
    axes[0].legend([], frameon=False)  # Remove legend for absolute counts

    # Relative counts
    rel_counts = df.groupby([cat1, cat2])[cat2].size() / df.groupby([cat1])[cat2].size()  # Calculate percentages
    rel_counts.unstack().plot.bar(stacked=True, ax=axes[1], color=colors)  # Plot stacked bar chart
    axes[1].set_title(f"{cat1.capitalize()} vs {cat2.capitalize()}, Relative Counts")
    axes[1].tick_params(axis="x", rotation=45)
    for label in axes[1].get_xticklabels():
        label.set_ha("right")
    axes[1].legend(loc=(1.01, 0))  # Place legend outside the plot

    # Adjust layout
    plt.tight_layout()
    plt.show()
    

def plot_countplot_with_hue(df, x_column, hue_column, title, xlabel, ylabel, palette, figsize, rotation, ha, legend_title, legend_bbox_to_anchor, legend_loc):
    """
    Create a countplot with hue.

    Parameters:
    - df (pd.DataFrame): The input dataframe.
    - x_column (str): The column for the x-axis.
    - hue_column (str): The column to use as the hue.
    - title (str): The title of the plot.
    - xlabel (str): Label for the x-axis.
    - ylabel (str): Label for the y-axis.
    - palette (str or list): The color palette for the plot.
    - figsize (tuple): The size of the figure (width, height).
    - rotation (int): Rotation angle for x-axis labels.
    - ha (str): Horizontal alignment for x-axis labels (default is 'right').
    - legend_title (str): Title for the legend.
    - legend_bbox_to_anchor (tuple): Position of the legend box (default is (1.05, 1)).
    - legend_loc (str): Location of the legend box (default is 'upper left').

    Returns:
    - None: Displays the plot.
    """
    # Set figure size
    plt.figure(figsize=figsize)
    
    # Create the countplot
    sns.countplot(data=df, x=x_column, hue=hue_column, palette=palette)
    
    # Customize legend
    plt.legend(title=legend_title, bbox_to_anchor=legend_bbox_to_anchor, loc=legend_loc)
    
    # Rotate x-axis labels
    plt.xticks(rotation=rotation, ha=ha)
    
    # Add title and axis labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    # Adjust layout to prevent clipping
    plt.tight_layout()
    
    # Show the plot
    plt.show()
    
    
# ----- 6.5.1  -----

def plot_aggregated_values_by_category(df, group_by_column, value_column, aggregation, title, xlabel, ylabel, palette, figsize, rotation):
    """
    Plot the aggregated values (mean or sum) of a numeric column grouped by a categorical column.

    Parameters:
    - df (pd.DataFrame): The input dataframe.
    - group_by_column (str): Column to group data by (e.g., 'last_promo').
    - value_column (str): Column with the numeric values to aggregate (e.g., 'product_count').
    - aggregation (str): Aggregation method - 'mean' or 'sum'.
    - title (str): Title for the plot.
    - xlabel (str): Label for the x-axis.
    - ylabel (str): Label for the y-axis.
    - palette (str): Color palette to use for the plot.
    - figsize (tuple): Size of the figure (width, height).
    - rotation (int): Rotation angle for x-axis labels.

    Returns:
    - None: Displays the bar plot.
    """
    # Validate the aggregation parameter
    if aggregation not in ['mean', 'sum']:
        raise ValueError("Invalid aggregation method. Choose 'mean' or 'sum'.")

    # Perform the aggregation
    if aggregation == 'mean':
        aggregated_data = df.groupby(group_by_column)[value_column].mean().reset_index()
    else:  # aggregation == 'sum'
        aggregated_data = df.groupby(group_by_column)[value_column].sum().reset_index()

    # Create the plot
    plt.figure(figsize=figsize)
    sns.barplot(
        x=group_by_column,
        y=value_column,
        data=aggregated_data,
        palette=palette
    )

    # Add labels and titles
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=rotation)

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()



def plot_grouped_bar_chart(df, group_by_column, value_columns, bar_width, title, xlabel, ylabel, colors, figsize, rotation):
    """
    Plot grouped bar chart for specified value columns.

    Parameters:
    - df (pd.DataFrame): The input dataframe.
    - group_by_column (str): The column to group data by (e.g., 'last_promo').
    - value_columns (list): List of columns to plot (e.g., ['tot_work_days', 'tot_leisure_days']).
    - bar_width (float): Width of the bars in the plot.
    - title (str): Title of the plot.
    - xlabel (str): Label for the x-axis.
    - ylabel (str): Label for the y-axis.
    - colors (list): List of colors for each group of bars (e.g., ['#66c2a5', '#fc8d62']).
    - figsize (tuple): Size of the figure (width, height).
    - rotation (int): Rotation angle for x-axis labels.

    Returns:
    - None: Displays the plot.
    """
    # Group and aggregate the data
    grouped_df = df.groupby(group_by_column)[value_columns].sum().reset_index()

    # Set bar positions
    r1 = range(len(grouped_df))
    r_positions = [r1]
    for i in range(1, len(value_columns)):
        r_positions.append([x + bar_width for x in r_positions[-1]])

    # Plot the bars
    plt.figure(figsize=figsize)
    for i, col in enumerate(value_columns):
        plt.bar(r_positions[i], grouped_df[col], color=colors[i], width=bar_width, label=col.replace('_', ' ').capitalize())

    # Add details to the chart
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks([r + bar_width * (len(value_columns) - 1) / 2 for r in r1], grouped_df[group_by_column], rotation=rotation)
    plt.legend()

    # Display the plot
    plt.tight_layout()
    plt.show()



def calculate_max_by_group(df, group_by_column, value_columns):
    """
    Calculate the maximum value for each group in the specified value columns.

    Parameters:
    - df (pd.DataFrame): The input dataframe.
    - group_by_column (str): The column to group data by (e.g., 'last_promo').
    - value_columns (list): List of columns to calculate maximum values (e.g., ['tot_work_days', 'tot_leisure_days']).

    Returns:
    - dict: Dictionary with the maximum category for each value column.
    """
    # Group and aggregate the data
    grouped_df = df.groupby(group_by_column)[value_columns].sum().reset_index()

    # Find the category with the maximum value for each column
    max_categories = {}
    for col in value_columns:
        max_category = grouped_df.loc[grouped_df[col].idxmax(), group_by_column]
        max_categories[col] = max_category

    return max_categories



# ----- 6.5.2  -----

def plot_boxplot(df, x_column, y_column, hue_column, palette, figsize, title, xlabel, ylabel, rotation):
    """
    Plot a boxplot for the specified columns with optional hue.

    Parameters:
    - df (pd.DataFrame): The input dataframe.
    - x_column (str): Column for the x-axis (e.g., 'last_promo').
    - y_column (str): Column for the y-axis (e.g., 'product_count').
    - hue_column (str, optional): Column to use as the hue (e.g., 'last_promo'). Default is None.
    - palette (str): Color palette to use for the boxplot (e.g., 'Set2'). Default is 'Set2'.
    - figsize (tuple): Figure size (width, height). Default is (12, 6).
    - title (str): Title of the plot. Default is an empty string.
    - xlabel (str): Label for the x-axis. Default is an empty string.
    - ylabel (str): Label for the y-axis. Default is an empty string.
    - rotation (int): Rotation angle for the x-axis labels. Default is 0.

    Returns:
    - None: Displays the boxplot.
    """
    plt.figure(figsize=figsize)
    sns.boxplot(x=x_column, y=y_column, data=df, hue=hue_column, palette=palette, legend=False)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=rotation)
    plt.tight_layout()
    plt.show()



def perform_anova(df, group_column, value_column):
    """
    Perform ANOVA test on the specified groups.

    Parameters:
    - df (pd.DataFrame): The input dataframe.
    - group_column (str): Column representing the groups (e.g., 'last_promo').
    - value_column (str): Column representing the values to compare (e.g., 'product_count').

    Returns:
    - tuple: F-statistic and p-value of the ANOVA test.
    """
    # Create a list of Series for each group
    groups = [df[df[group_column] == group][value_column] for group in df[group_column].unique()]
    
    # Perform ANOVA
    f_stat, p_value = f_oneway(*groups)
    
    return f_stat, p_value



def plot_boxplot_by_grouped_columns(df, columns, filter_column, y_column, figsize, palette, title, xlabel, ylabel, rotation=0):
    """
    Prepare data and plot a boxplot for specified grouped columns.

    Parameters:
    - df (pd.DataFrame): The input dataframe.
    - columns (list): List of columns to group and filter by (e.g., cuisines).
    - filter_column (str): Column to use for filtering (e.g., `customer_age`).
    - y_column (str): Name of the new column to store the group labels.
    - figsize (tuple): Size of the figure (width, height).
    - palette (str): Color palette to use for the plot.
    - title (str): Title of the plot.
    - xlabel (str): Label for the x-axis.
    - ylabel (str): Label for the y-axis.
    - rotation (int): Rotation angle for x-axis labels (default is 0).

    Returns:
    - None: Displays the boxplot.
    """
    # Prepare an empty list for data
    boxplot_data = []

    # Iterate over the specified columns
    for column in columns:
        # Filter rows where the column has values greater than 0
        filtered_data = df[df[column] > 0][[filter_column]].copy()
        filtered_data[y_column] = column  # Add the column name as a label
        boxplot_data.append(filtered_data)  # Append to the list

    # Concatenate all data into a single DataFrame
    concatenated_data = pd.concat(boxplot_data, ignore_index=True)

    # Plot the boxplot
    plt.figure(figsize=figsize)
    sns.boxplot(x=y_column, y=filter_column, data=concatenated_data, palette=palette)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=rotation)
    plt.tight_layout()
    plt.show()

      
# =============================
# Section 7
# =============================


# ----- 7.1  -----
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

# ----- 7.2.1  -----

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

# ----- 7.2.2  -----
def clean_and_fill_with_mode(df, column, replace_values, mode_fill=True):
    """
    Replace specified values with NaN and optionally fill missing values with the column's mode.

    Parameters:
    - df (pd.DataFrame): The input dataframe.
    - column (str): The column to clean and fill.
    - replace_values (list): List of values to replace with NaN.
    - mode_fill (bool): If True, fill missing values with the column's mode. Default is True.

    Returns:
    - pd.DataFrame: The modified dataframe with updated column.
    """
    # Replace specified values with NaN
    df[column] = df[column].replace(replace_values, np.nan)
    
    # Optionally fill missing values with the column's mode
    if mode_fill:
        mode_value = df[column].mode()[0]
        df[column] = df[column].fillna(mode_value)
    
    return df



# ----- 7.3.1  -----

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




# ----- 7.3.2  -----

def apply_manual_filters(df):
    """
    Apply manual filters to remove extreme outliers from the dataset.

    Parameters:
    - df (pd.DataFrame): The input DataFrame to filter.

    Returns:
    - pd.DataFrame: A filtered DataFrame with rows that satisfy the conditions.
    """
    filters = (
        (df['vendor_count'] <= 30) &
        (df['product_count'] <= 100) &
        (df['tot_CUI'] <= 500) &
        (df['tot_work_days'] <= 30) &
        (df['tot_leisure_days'] <= 30) &
        (df['total_products_by_week'] <= 50) &
        (df['total_products_by_day'] <= 50) &
        (df['tot_early_morning'] <= 20) &
        (df['tot_breakfast'] <= 25) &
        (df['tot_lunch'] <= 20) &
        (df['tot_afternoon'] <= 20) &
        (df['tot_dinner'] <= 20) &
        (df['tot_late_night'] <= 8) &
        (df['tot_western_cuisines'] <= 250) &
        (df['tot_oriental_cuisines'] <= 300) &
        (df['tot_other_cuisines'] <= 150) &
        (df['purchased_cuisines'] <= 6)
    )
    
    # Apply the filters and return the filtered DataFrame
    return df[filters]


# ----- 7.4.2  -----


def plot_cramers_v_heatmap(df, categorical_features, figsize, cmap, annot):
    """
    Calculate and plot a Cramér's V heatmap for a set of categorical features.

    Parameters:
    - df (pd.DataFrame): The input dataframe containing the categorical features.
    - categorical_features (list): List of categorical feature names to analyze.
    - figsize (tuple): Figure size for the heatmap.
    - cmap (str): Color map for the heatmap.
    - annot (bool): Whether to annotate the heatmap with values.

    Returns:
    - None: Displays the heatmap.
    """
    # Cramér's V matrix
    cramers_v_matrix = pd.DataFrame(index=categorical_features, columns=categorical_features)

    # Calculate Cramér's V for each pair
    for i in range(len(categorical_features)):
        for j in range(i, len(categorical_features)):
            if i == j:
                cramers_v_matrix.iloc[i, j] = 1.0
            else:
                cramers_v_matrix.iloc[i, j] = cramers_v(
                    df[categorical_features[i]], df[categorical_features[j]]
                )

    # Convert to float to avoid plotting issues
    cramers_v_matrix = cramers_v_matrix.astype(float)

    # Plot the heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(cramers_v_matrix, annot=annot, cmap=cmap, square=True, linewidths=.5, cbar_kws={'shrink': .8})
    plt.title("Cramér's V Heatmap for Categorical Features")
    plt.show()


# ----- 7.5 -----

def plot_scaled_boxplots(df_original, df_minmax, df_standard, features, figsize):
    """
    Plot boxplots for original, MinMax scaled, and Standard scaled versions of numeric features.

    Parameters:
    - df_original (pd.DataFrame): DataFrame with the original data.
    - df_minmax (pd.DataFrame): DataFrame with MinMax scaled data.
    - df_standard (pd.DataFrame): DataFrame with Standard scaled data.
    - features (list): List of numeric features to plot.
    - figsize (tuple): Size of the figure (width, height).

    Returns:
    - None: Displays the boxplots.
    """
    sns.set_style('whitegrid')
    
    # Create subplots
    fig, axes = plt.subplots(len(features), 3, figsize=figsize, 
                             tight_layout=True, sharex=False, sharey=False)
    
    # Iterate over numeric features
    for i, feature in enumerate(features):
        sns.boxplot(data=df_original, x=feature, ax=axes[i][0], width=0.4)
        axes[i][0].set_title('Original')
        axes[i][0].set_ylabel(feature)

        sns.boxplot(data=df_minmax, x=feature, ax=axes[i][1], width=0.4)
        axes[i][1].set_title('MinMaxScaler()')

        sns.boxplot(data=df_standard, x=feature, ax=axes[i][2], width=0.4)
        axes[i][2].set_title('StandardScaler()')

        # Optionally, remove x-axis labels for a cleaner look
        axes[i][0].set_xlabel(None)
        axes[i][1].set_xlabel(None)
        axes[i][2].set_xlabel(None)
    plt.show()



def plot_scaled_histograms(df_original, df_minmax, df_standard, features, bins=15, figsize=(5, 5)):
    """
    Plot histograms for original, MinMax scaled, and Standard scaled versions of numeric features.

    Parameters:
    - df_original (pd.DataFrame): DataFrame with the original data.
    - df_minmax (pd.DataFrame): DataFrame with MinMax scaled data.
    - df_standard (pd.DataFrame): DataFrame with Standard scaled data.
    - features (list): List of numeric features to plot.
    - bins (int): Number of bins for the histograms.
    - figsize (tuple): Size of the figure for each feature (width, height).

    Returns:
    - None: Displays the histograms for each feature.
    """
    sns.set_style('whitegrid')

    # Iterate over numeric features
    for feature in features:
        # Create subplots with 3 rows, 1 column for each feature
        fig, axes = plt.subplots(3, 1, figsize=figsize, tight_layout=True)

        # Common histogram arguments
        hp_args = dict(x=feature, bins=bins)

        # Original Data
        sns.histplot(df_original, ax=axes[0], **hp_args)
        axes[0].set_title(f'{feature}: Original')
        axes[0].set_xlabel(None)

        # MinMaxScaler Data
        sns.histplot(df_minmax, ax=axes[1], **hp_args)
        axes[1].set_title(f'{feature}: MinMaxScaler()')
        axes[1].set_xlabel(None)

        # StandardScaler Data
        sns.histplot(df_standard, ax=axes[2], **hp_args)
        axes[2].set_title(f'{feature}: StandardScaler()')
        axes[2].set_xlabel(None)

        # Display the histograms for the current feature
        plt.show()
