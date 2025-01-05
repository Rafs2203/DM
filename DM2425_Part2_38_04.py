# IMPORTS -----------------------------------------------------------------




#organizar:


import pandas as pd  # Imports the pandas library, used for handling and analyzing structured data in tables (DataFrames).
import numpy as np  # Imports numpy, which is helpful for working with numerical data, especially for mathematical functions and arrays.
import matplotlib.pyplot as plt  # Imports matplotlib's pyplot module to create plots and charts for visualizing data.
import seaborn as sns  # Imports seaborn, a library built on matplotlib, for creating more advanced and visually appealing statistical graphics.
from math import ceil  # Imports the ceil function from the math library, which rounds numbers up to the nearest whole number.
from scipy.stats import f_oneway # Import the f_oneway function from SciPy for performing a one-way ANOVA test
import matplotlib.cm as cm
import matplotlib.colors as mpl_colors
from matplotlib.patches import RegularPolygon
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colorbar
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score


# --
from os.path import join
from scipy.cluster.hierarchy import dendrogram
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import  AgglomerativeClustering, DBSCAN, KMeans, MeanShift, estimate_bandwidth
from minisom import MiniSom 
from sklearn.metrics import silhouette_score, silhouette_samples


sns.set()










############################################################################################################
#--------------------------- Functions used in Clustering aka DM2425_Part2_38_02 ---------------------------
############################################################################################################


# =============================
# Section 2
# =============================

def get_sst(df_clusters, feats):
    """
    Calculate the sum of squares (SST) for the given DataFrame.

    The sum of squares is computed as the sum of the variances of each column
    multiplied by the number of non-NA/null observations minus one.

    Parameters:
    df_clusters (pandas.DataFrame): The input DataFrame for which the sum of squares is to be calculated.
    feats (list of str): A list of feature column names to be used in the calculation.

    Returns:
    float: The sum of squares of the DataFrame.
    """
    df_clusters_ = df_clusters[feats]
    sst = np.sum(df_clusters_.var() * (df_clusters_.count() - 1))
    
    return sst 



def get_ssb(df_clusters, feats, label_col):
    """
    Calculate the between-group sum of squares (SSB) for the given DataFrame.
    The between-group sum of squares is computed as the sum of the squared differences
    between the mean of each group and the overall mean, weighted by the number of observations
    in each group.

    Parameters:
    df_clusters (pandas.DataFrame): The input DataFrame containing the data.
    feats (list of str): A list of feature column names to be used in the calculation.
    label_col (str): The name of the column in the DataFrame that contains the group labels.
    
    Returns
    float: The between-group sum of squares of the DataFrame.
    """
    
    ssb_i = 0
    for i in np.unique(df_clusters[label_col]):
        df_clusters_ = df_clusters.loc[:, feats]
        X_ = df_clusters_.values
        X_k = df_clusters_.loc[df_clusters[label_col] == i].values
        
        ssb_i += (X_k.shape[0] * (np.square(X_k.mean(axis=0) - X_.mean(axis=0))) )

    ssb = np.sum(ssb_i)
    

    return ssb



def get_ssw(df_clusters, feats, label_col):
    """
    Calculate the sum of squared within-cluster distances (SSW) for a given DataFrame.

    Parameters:
    df_clusters (pandas.DataFrame): The input DataFrame containing the data.
    feats (list of str): A list of feature column names to be used in the calculation.
    label_col (str): The name of the column containing cluster labels.

    Returns:
    float: The sum of squared within-cluster distances (SSW).
    """
    feats_label = feats+[label_col]

    df_clusters_k = df_clusters[feats_label].groupby(by=label_col).apply(lambda col: get_sst(col, feats), 
                                                       include_groups=False)

    return df_clusters_k.sum()



def get_rsq(df_clusters, feats, label_col):
    """
    Calculate the R-squared value for a given DataFrame and features.

    Parameters:
    df_clusters (pd.DataFrame): The input DataFrame containing the data.
    feats (list): A list of feature column names to be used in the calculation.
    label_col (str): The name of the column containing the labels or cluster assignments.

    Returns:
    float: The R-squared value, representing the proportion of variance explained by the clustering.
    """

    df_clusters_sst_ = get_sst(df_clusters, feats)                 # get total sum of squares
    df_clusters_ssw_ = get_ssw(df_clusters, feats, label_col)      # get ss within
    df_clusters_ssb_ = df_clusters_sst_ - df_clusters_ssw_         # get ss between

    # r2 = ssb/sst 
    return (df_clusters_ssb_/df_clusters_sst_)



def get_r2_hc(df_clusters, link_method, max_nclus, min_nclus=1, dist="euclidean"):
    """This function computes the R2 for a set of cluster solutions given by the application of a hierarchical method.
    The R2 is a measure of the homogenity of a cluster solution. It is based on SSt = SSw + SSb and R2 = SSb/SSt. 
    
    Parameters:
    df_clusters (DataFrame): Dataset to apply clustering
    link_method (str): either "ward", "complete", "average", "single"
    max_nclus (int): maximum number of clusters to compare the methods
    min_nclus (int): minimum number of clusters to compare the methods. Defaults to 1.
    dist (str): distance to use to compute the clustering solution. Must be a valid distance. Defaults to "euclidean".
    
    Returns:
    ndarray: R2 values for the range of cluster solutions
    """
    
    r2 = []  # where we will store the R2 metrics for each cluster solution
    feats = df_clusters.columns.tolist()
    
    for i in range(min_nclus, max_nclus+1):  # iterate over desired ncluster range
        cluster = AgglomerativeClustering(n_clusters=i, metric=dist, linkage=link_method)
        
        #get cluster labels
        hclabels = cluster.fit_predict(df_clusters) 
        
        # concat df_clusters with labels
        df_clusters_concat = pd.concat([df_clusters, pd.Series(hclabels, name='labels', index=df_clusters.index)], axis=1)  
        
        
        # append the R2 of the given cluster solution
        r2.append(get_rsq(df_clusters_concat, feats, 'labels'))
        
    return np.array(r2)



# =============================
# Section 3
# =============================


# ----- 3.1 -----
def inertia(df, start, end, step):
    """
    Calculate and visualize the inertia for K-Means clustering with different numbers of clusters.

    This function computes the inertia (within-cluster sum of squared distances) for a range of cluster counts 
    using the K-Means algorithm. It helps determine the optimal number of clusters by plotting the 
    "elbow curve," which shows how inertia changes as the number of clusters increases.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing the data to cluster.
    - start (int): The starting number of clusters to evaluate (inclusive).
    - end (int): The ending number of clusters to evaluate (exclusive).
    - step (int): The step size for the range of cluster numbers to evaluate.

    Returns:
    - None: Displays a plot of inertia values for different numbers of clusters.
    """
    # Define the range of cluster numbers to evaluate
    k_values = range(start, end, step)
    
    # List to store the inertia values
    inertias = []
    
    # Iterate over the range of cluster numbers
    for k in k_values:
        # Apply K-Means clustering
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(df)  
        
        # Store the inertia value for the current k
        inertias.append(kmeans.inertia_)
    
    # Plot the inertia values
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, inertias, marker='o', linestyle='-', color='b')
    plt.title('K-Means Inertia for Different Values of k')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.grid(True)
    plt.show()



def calculate_max_clusters(df):
    """
    Calculate the maximum number of possible clusters in a dataset by counting unique rows.

    This function identifies the number of unique data points in the given DataFrame, 
    which represents the upper limit for the number of clusters that can be formed.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.

    Returns:
    - None: Print the number of unique data points, which represents the maximum number of possible clusters.
    """
    unique_points = pd.DataFrame(df).drop_duplicates().shape[0]
    print(f"Maximum number of possible clusters: {unique_points}")

    

def apply_kmeans_clustering(df, n_clusters, init='k-means++', n_init=15, random_state=42):
    """
    Apply K-Means clustering to a given DataFrame and return the grouped cluster means.

    Parameters:
    - df (pd.DataFrame): The input DataFrame to cluster.
    - n_clusters (int): The number of clusters to form.
    - init (str): Initialization method for K-Means. Default is 'k-means++'.
    - n_init (int): Number of times the algorithm will run with different centroid seeds. Default is 15.
    - random_state (int): Random state for reproducibility. Default is 42.

    Returns:
    - pd.DataFrame: A DataFrame grouped by the K-Means cluster labels with mean values for each cluster.
    - np.ndarray: Cluster centers from the K-Means algorithm.
    """
    # Initialize the KMeans algorithm with specified parameters
    kmeans = KMeans(n_clusters=n_clusters, init=init, n_init=n_init, random_state=random_state)

    # Fit the KMeans algorithm and predict cluster labels
    km_labels = kmeans.fit_predict(df)

    # Extract the cluster centers
    clust_centers = kmeans.cluster_centers_

    # Create a copy of the input DataFrame and add the cluster labels
    df_kmeans = df.copy()
    df_kmeans['kmeans_labels'] = km_labels

    # Group by the 'kmeans_labels' column and calculate the mean for each cluster
    df_kmeans = df_kmeans.groupby('kmeans_labels').mean()

    return df_kmeans, clust_centers, km_labels



def plot_r2_hc_methods(r2_hc_df, max_nclus, title="$R^2$ plot for various hierarchical methods", 
                       figsize=(11, 5), linewidth=2.5, marker="o", xlabel="Number of clusters", 
                       ylabel="R2 metric", legend_title="HC methods", legend_fontsize=11, 
                       xticks_fontsize=11, title_fontsize=21):
    """
    Plot a line graph for R² metrics across different hierarchical clustering methods.

    Parameters:
    - r2_hc_df (pd.DataFrame): DataFrame containing R² values for each method and number of clusters.
    - max_nclus (int): Maximum number of clusters used in the evaluation.
    - title (str): Title for the plot. Default is "$R^2$ plot for various hierarchical methods".
    - figsize (tuple): Size of the figure. Default is (11, 5).
    - linewidth (float): Line width for the plot. Default is 2.5.
    - marker (str): Marker style for the lines. Default is "o".
    - xlabel (str): Label for the x-axis. Default is "Number of clusters".
    - ylabel (str): Label for the y-axis. Default is "R2 metric".
    - legend_title (str): Title for the legend. Default is "HC methods".
    - legend_fontsize (int): Font size for the legend title. Default is 11.
    - xticks_fontsize (int): Font size for x-axis ticks. Default is 11.
    - title_fontsize (int): Font size for the plot title. Default is 21.

    Returns:
    - None: Displays the plot.
    """
    # Create the plot
    fig = plt.figure(figsize=figsize)
    sns.lineplot(data=r2_hc_df, linewidth=linewidth, markers=[marker] * len(r2_hc_df.columns))

    # Finalize the plot
    plt.legend(title=legend_title, title_fontsize=legend_fontsize)
    plt.xticks(range(1, max_nclus + 1), fontsize=xticks_fontsize)
    plt.xlabel(xlabel, fontsize=13)
    plt.ylabel(ylabel, fontsize=13)

    # Add the title
    fig.suptitle(title, fontsize=title_fontsize)

    # Display the plot
    plt.show()



def compute_linkage_matrix(hclust):
    """
    Compute the linkage matrix for hierarchical clustering from an AgglomerativeClustering model.

    This function creates a linkage matrix used for plotting dendrograms by leveraging 
    the children_, distances_, and cluster size information from the AgglomerativeClustering model.

    Parameters:
    - hclust (AgglomerativeClustering): A fitted AgglomerativeClustering model with `distance_threshold=0` and `n_clusters=None`.

    Returns:
    - np.ndarray: A linkage matrix with the following columns:
        - Column 1: ID of the first cluster/observation being merged.
        - Column 2: ID of the second cluster/observation being merged.
        - Column 3: Distance between the two clusters/observations.
        - Column 4: Number of points in the merged cluster.
    """
    # create the counts of samples under each node (number of points being merged)
    counts = np.zeros(hclust.children_.shape[0])
    n_samples = len(hclust.labels_)

    # hclust.children_ contains the observation ids that are being merged together
    # At the i-th iteration, children[i][0] and children[i][1] are merged to form node n_samples + i
    for i, merge in enumerate(hclust.children_):
        # track the number of observations in the current cluster being formed
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                # If this is True, then we are merging an observation
                current_count += 1  # leaf node
            else:
                # Otherwise, we are merging a previously formed cluster
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    # the hclust.children_ is used to indicate the two points/clusters being merged (dendrogram's u-joins)
    # the hclust.distances_ indicates the distance between the two points/clusters (height of the u-joins)
    # the counts indicate the number of points being merged (dendrogram's x-axis)
    linkage_matrix = np.column_stack(
        [hclust.children_, hclust.distances_, counts]
    ).astype(float)

    return linkage_matrix


def plot_dendrogram(linkage_matrix, linkage_name, distance_metric, y_threshold, figsize=(11, 5), p=5, above_threshold_color='k'):
    """
    Plot a dendrogram for hierarchical clustering using a linkage matrix.

    Parameters:
    - linkage_matrix (np.ndarray): The linkage matrix (e.g., from compute_linkage_matrix).
    - linkage_name (str): The name of the linkage method (e.g., "ward", "complete").
    - distance_metric (str): The distance metric used (e.g., "euclidean", "manhattan").
    - y_threshold (float): The horizontal threshold line for distance.
    - figsize (tuple): The size of the figure (width, height). Default is (11, 5).
    - p (int): The number of levels to truncate the dendrogram. Default is 5.
    - above_threshold_color (str): Color for the clusters above the threshold. Default is 'k' (black).

    Returns:
    - None: Displays the dendrogram plot.
    """

    sns.set()  # Set Seaborn's default style

    # Create the plot
    fig = plt.figure(figsize=figsize)

    # Plot the dendrogram
    dendrogram(
        linkage_matrix,
        truncate_mode='level',
        p=p,
        color_threshold=y_threshold,
        above_threshold_color=above_threshold_color,
    )

    # Add the horizontal threshold line
    plt.hlines(y_threshold, 0, plt.xlim()[1], colors="r", linestyles="dashed")

    # Add titles and labels
    plt.title(f'Hierarchical Clustering Dendrogram: {linkage_name.title()} Linkage', fontsize=21)
    plt.xlabel('Number of points in node (or index of point if no parenthesis)')
    plt.ylabel(f'{distance_metric.title()} Distance', fontsize=13)

    # Show the plot
    plt.show()
    
# =============================
# Section 4
# =============================

def characterize_clusters(group_features, cluster_labels, df):
    """
    Characterize clusters by grouping data based on cluster labels and calculating 
    the mean of specified features for each cluster.

    Parameters:
    - group_features (list): List of feature column names to include in the analysis.
    - cluster_labels (array-like): Cluster labels assigned to each data point.
    - df (DataFrame): Original DataFrame containing the features.

    Returns:
    - DataFrame: A DataFrame with the mean values of the specified features for each cluster.
    """
    # Concatenate the cluster labels with the original features
    df_concat = pd.concat((df[group_features], pd.Series(cluster_labels, name='labels', index=df.index)), axis=1)

    print("BD index for this solution: ",davies_bouldin_score(df[group_features], cluster_labels))
    print("Silhouette score for this solution: ",silhouette_score(df[group_features], cluster_labels))
    
    # Group by the cluster label and calculate the mean of each feature for each cluster
    return df_concat.groupby('labels').mean()



def plot_inertia_for_categories(categories, df, Customer_profile):
    """
    Function to plot inertia values for KMeans clustering for each category.

    Parameters:
    - categories: A dictionary where the key is the category name and the value is the list of features.
    - df: The dataframe containing the customer data.
    - Customer_profile: The column name in the dataframe that contains customer profile data.

    This function generates a plot of inertia values for each category, helping to determine
    the optimal number of clusters by showing the "elbow" point.
    """
    
    # Loop through the items in the 'categories' dictionary
    for category, features in categories.items():
        # Initialize an empty list to store the inertia values for each clustering
        inertia = []
        
        # Define the range of cluster numbers to test (from 1 to 10)
        range_clusters = range(1, 11)
        
        # Iterate over the specified range of cluster numbers (n_clusters)
        for n_clus in range_clusters:
            # Initialize the KMeans clustering with the current number of clusters
            kmclust = KMeans(n_clusters=n_clus, init='random', n_init=15, random_state=1)
            
            # Fit the KMeans model to the data for the current category
            kmclust.fit(df[Customer_profile])
            
            # Append the inertia value (sum of squared distances of samples to their centroids)
            # for the current clustering solution to the inertia list
            inertia.append(kmclust.inertia_)
        
        # Create a plot to visualize the inertia values across different cluster numbers
        fig, ax = plt.subplots(figsize=(9, 5))
        
        # Plot the inertia values against the range of cluster numbers
        ax.plot(range_clusters, inertia)
        
        # Set the x-axis ticks to correspond to the cluster numbers
        ax.set_xticks(range_clusters)
        
        # Label the y-axis as "Inertia: SSw" (sum of squared within-cluster distances)
        ax.set_ylabel("Inertia: SSw")
        
        # Label the x-axis as "Number of clusters"
        ax.set_xlabel("Number of clusters")
        
        # Set the title of the plot to include the category name
        ax.set_title(f"Inertia plot over clusters for {category}", size=15)

        # Display the plot
        plt.show()

