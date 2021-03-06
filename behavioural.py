# %%
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.cluster import KMeans
from tqdm import tqdm

# %%
COLUMNS_LIST = ["auction.id", "auction.start_date", "auction.end_date",
                "auction.lot_min_start_date", "auction.lot_max_start_date",
                "auction.lot_min_end_date", "auction.lot_max_end_date", "lot.id",
                "lot.closingdate", "lot.startingdate",
                "lot.valid_bid_count", "lot.starting_at_1EUR", "lot.is_sold", "bid.id",
                "user.id", "bid.is_autobid", "bid.is_valid", "bid.amount",
                "bid.is_latest", "bid.is_first", "bid.days_to_close", "bid.date",
                "bid.added_bidvalue", "lot.start_amount"]

# %%
df = pd.read_csv(
    "data/data_merged.csv.gz",
    usecols= COLUMNS_LIST,
    parse_dates=[
        "auction.start_date",
        "auction.end_date",
        "auction.lot_min_start_date",
        "auction.lot_max_start_date",
        "auction.lot_min_end_date",
        "auction.lot_max_end_date",
        "lot.closingdate",
        "lot.startingdate",
        "bid.date",
    ],
)

# %%
# Test that all bids in the dataset are valid
print(f"Unique values for bid.is_valid are {df['bid.is_valid'].unique()}")
# %% Drop invalid bids
df = df[df["bid.is_valid"] != 0] 

# %%
def create_cluster_datav2(df: pd.DataFrame, file_name: str) -> pd.DataFrame:
    """Function which preprocesses data to create features for clustering

    Args:
        df (pd.DataFrame): input dataframe with auction data
        file_name (str): File name to which to write the new dataframe

    Returns:
        pd.DataFrame: Dataframe containing clustering features
    """
    
    #Create dictionary to store results for new clustering dataframe
    cluster_dict = {"lot_user_combi": [],
                    "nr_bids": [],
                    "time_of_entry": [],
                    "time_of_exit": [],
                    "bid_increment": [],
                    "autobid": []}
    
    
    for lot_id in tqdm(df["lot.id"].unique(), desc= "Creating clustering features"):
        df_this_lot = df[df["lot.id"] == lot_id]
        # Check if valid_bid_count is equal to the number of bids in the data for this lot 
        if df_this_lot["lot.valid_bid_count"].mean() == df_this_lot.shape[0]:
            # Iterate over each user that made a bid for this lot
            for user_id in df[df["lot.id"] == lot_id]["user.id"].unique():
                if not np.isnan(user_id):
                    #For this lot, user combi, which has a bidding behaviour, create features
                    df_user_lot = df[(df["lot.id"] == lot_id) & (df["user.id"] == user_id)]
                    # Store the lot-user combination
                    cluster_dict["lot_user_combi"].append(str((lot_id, user_id)))
                    
                    ### Find the number of bids by this user ###
                    # We denote this nr_of_bids_this_user / total_nr_bids on this lot
                    cluster_dict["nr_bids"].append(df_user_lot.shape[0])
                    # print(df_user_lot)
                    ### Find the scaled time of entry ###
                    # Sort values by bidding time in chronological order
                    df_user_lot = df_user_lot.sort_values(
                        by="bid.date").reset_index(drop=True)
                    #Find the moment of the first bid in hours after lot startingmoment
                    # -> We use hours, since bid moments are all given on exact hours
                    #Find the entry time in hours                  
                    entry_time_in_hours = (df_user_lot["bid.date"][0] - df_user_lot["auction.lot_min_start_date"][0]) / pd.Timedelta("1 hour")
                    total_time_auction = (df_user_lot.iloc[0]["auction.lot_max_end_date"] - df_user_lot.iloc[0]["auction.lot_min_start_date"]) / pd.Timedelta("1 hour")
                    #Divide the entry time by the total time of the lot
                    time_of_entry_fraction = (entry_time_in_hours / total_time_auction) * 5 + 1
                    cluster_dict["time_of_entry"].append(time_of_entry_fraction)

                    ### Find the time of exit ###
                    
                    exit_time_in_hours = (df_user_lot["bid.date"][df_user_lot.shape[0] - 1] - df_user_lot["auction.lot_min_start_date"][df_user_lot.shape[0] - 1]) / pd.Timedelta("1 hour")
                    exit_time_fraction = (exit_time_in_hours / total_time_auction) * 5 + 1
                    cluster_dict["time_of_exit"].append(exit_time_fraction)
                    
                    ### Find the average bid increment ###
                    
                    
                    # Sort the values of the lot in chronological order
                    df_this_lot_sorted = df_this_lot.sort_values(by=["bid.date", "bid.amount"]).reset_index(drop=True)
                    # Find the winning/end price of this lot
                    end_price = df_this_lot_sorted[df_this_lot_sorted["bid.is_latest"] == 1]["bid.amount"]
                    #Iterate over each row
                    increment_list = []
                    for index in range(df_this_lot_sorted.shape[0]):
                        #If the row is of a bid of the current user, calculate the increment
                        if df_this_lot_sorted.iloc[index]["user.id"] == user_id:
                            #If this is the  first bid
                            if index == 0:
                                #Increment = bid amount - starting price
                                increment = (df_this_lot_sorted.iloc[index]["bid.amount"] -
                                            df_this_lot_sorted.iloc[index]["lot.start_amount"]) / end_price
                            #If it is not the first bid
                            else:
                                #Increment = bid_amount / previous bid_amount
                                increment = (df_this_lot_sorted.iloc[index]["bid.amount"] -
                                            df_this_lot_sorted.iloc[index - 1]["bid.amount"]) / end_price
                                
                            increment_list.append(increment)
                    #Append the mean increment number to the cluster_dict
                    cluster_dict["bid_increment"].append(np.array(increment_list).mean() * 5 + 1)
                    ## Check for autobid
                    #Append the percentage of bids in this auction by the user that were made using autobid
                    cluster_dict["autobid"].append(df_user_lot["bid.is_autobid"].mean() * 5 + 1)

                
                
                
    #Transform the cluster_dict to a Pandas DataFrame
    cluster_df = pd.DataFrame(cluster_dict) 
    if not os.path.exists("Data"):
        os.makedirs("Data")
    cluster_df.to_csv(os.path.join("Data", file_name), index = False)
    
    return cluster_df

# %% Perform data preprocessing (Can skip this, as we have already saved this data!!!!!!)
cluster_data_v2 = create_cluster_datav2(df = df, file_name= "cluster_features_v2.csv")

# %%
cluster_data_v2 = pd.read_csv(os.path.join("Data", "cluster_features_v2.csv"))

# %% Perform clustering for v2
silhouette_scores_list_v2 = []
davies_bouldin_list_v2 = []
inertia_list_v2 = []
model_list_v2 = []
for nr_clusters in tqdm(range(2, 10), desc = "Performing clustering for multiple k-values"):
    X = cluster_data_v2.drop("lot_user_combi", axis=1)
    kmeans_model = KMeans(n_clusters = nr_clusters, random_state = 0).fit(X)
    labels = kmeans_model.labels_
    silhouette_scores_list_v2.append(metrics.silhouette_score(X, labels, metric='euclidean'))
    davies_bouldin_list_v2.append(metrics.davies_bouldin_score(X, labels))
    inertia_list_v2.append(kmeans_model.inertia_)
    model_list_v2.append(kmeans_model)
# Plot performance measures
plt.plot(list(range(2, 10)), silhouette_scores_list_v2)
plt.show()
plt.plot(list(range(2, 10)), davies_bouldin_list_v2)
plt.show()
plt.plot(list(range(2, 10)), inertia_list_v2)
plt.xlabel("Number of Clusters")
plt.ylabel("Sum of Squares")


# %% # Print the cluster centers of the different cluster models
print("Model with k = 3:", model_list_v2[1].cluster_centers_, sep = "\n")
print("Model with k = 4:", model_list_v2[2].cluster_centers_, sep = "\n")
print("Model with k = 5:", model_list_v2[3].cluster_centers_, sep = "\n")
print("Model with k = 6:", model_list_v2[4].cluster_centers_, sep = "\n")
# %% Perform hierarchical clustering

###This caused error due to too much RAM being needed, so don't run this
# import time
# from sklearn.cluster import AgglomerativeClustering
# now = time.time()
# X = cluster_data_v2.drop("lot_user_combi", axis=1)
# clustering = AgglomerativeClustering().fit(X)
# end = time.time()
# print(f"Total time hierarchical clustering is {end - now} seconds")
# %%


