# IMPORTS -----------------------------------------------------------------

import pandas as pd  # Imports the pandas library, used for handling and analyzing structured data in tables (DataFrames).
import numpy as np  # Imports numpy, which is helpful for working with numerical data, especially for mathematical functions and arrays.
import matplotlib.pyplot as plt  # Imports matplotlib's pyplot module to create plots and charts for visualizing data.
import seaborn as sns  # Imports seaborn, a library built on matplotlib, for creating more advanced and visually appealing statistical graphics.
from math import ceil  # Imports the ceil function from the math library, which rounds numbers up to the nearest whole number.
from scipy.stats import f_oneway # Import the f_oneway function from SciPy for performing a one-way ANOVA test
from os.path import join
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram

sns.set()






# editar estes imports


##############################
# Data Manipulation & Utilities
##############################
import os
import time
import math
import sqlite3
import datetime
import textwrap
from itertools import combinations
from collections import Counter
from dateutil.relativedelta import relativedelta

##############################
# Core Scientific & Numeric Libraries
##############################
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency

##############################
# Visualization Libraries
##############################
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from matplotlib.colors import ListedColormap

##############################
# Machine Learning Preprocessing
##############################
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import (
    MinMaxScaler, 
    StandardScaler, 
    RobustScaler, 
    OrdinalEncoder, 
    LabelEncoder, 
    OneHotEncoder
)
from sklearn.decomposition import PCA
from imblearn.under_sampling import NearMiss, TomekLinks, ClusterCentroids

##############################
# Feature Selection
##############################
from sklearn.feature_selection import RFE, SelectKBest, f_classif, chi2

##############################
# Model Building
##############################
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier , RandomForestRegressor
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier


##############################
# Model Evaluation & Selection
##############################
from sklearn.model_selection import (
    train_test_split, 
    GridSearchCV, 
    RandomizedSearchCV, 
    cross_val_score, 
    StratifiedKFold,
    PredefinedSplit
)
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    precision_score, 
    recall_score, 
    roc_auc_score, 
    classification_report, 
    make_scorer, 
    precision_recall_curve, 
    roc_curve, 
    auc
)

##############################
# Pipeline & Column Transformations
##############################
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

##############################
# Data Balancing & Sampling
##############################
from imblearn.over_sampling import ADASYN, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN, SMOTETomek

##############################
# Misc
##############################
import warnings




# FUNCTIONS ---------------------------------------------------------------
### EDA -------------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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


##############################
# 4.3.3
##############################

# Grouping the less frequent chains to a category: "Other Chains"
def group_rare_values(df, column_name, threshold, name_category='Other Chains'):
    """
    Group rare values in a specified column of a DataFrame into a new category.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - column_name (str): The name of the column to analyze.
    - threshold (int): The frequency threshold below which values are grouped.
    - other_category (str): The name of the category for rare values. Default is 'Other Chains'.

    Returns:
    - pd.DataFrame: The DataFrame with the modified column.
    """
    # Calculate the frequency of unique values in the specified column
    value_frequency = df[column_name].value_counts()

    # Identify values that appear less frequently than the threshold
    rare_values = value_frequency[value_frequency < threshold].index

    # Replace rare values with a new category
    df[column_name] = df[column_name].apply(lambda x: name_category if x in rare_values else x)

    return df




















