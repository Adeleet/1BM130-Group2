# %%
import pandas as pd
from tqdm import tqdm
# %%
df = pd.read_csv(
    "data/data_merged.csv.gz",
    parse_dates=[
        "auction.start_date",
        "auction.end_date",
        "auction.lot_min_start_date",
        "auction.lot_max_start_date",
        "auction.lot_min_end_date",
        "auction.lot_max_end_date",
        "lot.closingdate",
        "lot.startingdate",
        "lot.collection_date",
        "bid.date",
    ],
)

# %% #Select columns that may potentionally be relevant
df = df[["auction.id", "auction.start_date", "auction.end_date",
         "auction.lot_min_start_date", "auction.lot_max_start_date",
         "auction.lot_min_end_date","auction.lot_max_end_date", "lot.id",
         "lot.closingdate", "lot.startingdate",
         "lot.valid_bid_count", "lot.starting_at_1EUR", "lot.is_sold", "bid.id",
         "user.id", "bid.is_autobid", "bid.is_valid", "bid.amount",
         "bid.is_latest", "bid.is_first", "bid.days_to_close", "bid.date",
         "bid.added_bidvalue", "lot.start_amount"]]

# %%
# Test that all bids in the dataset are valid
print(f"Unique values for bid.is_valid are {df['bid.is_valid'].unique()}")
# %% Drop invalid bids
df = df[df["bid.is_valid"] != 0] 
# %% Remove this cell later
df = df.iloc[:500]

# %%
#Create dictionary to store results for new clustering dataframe
cluster_dict = {"lot_user_combi": [],
                "nr_bids": [],
                "time_of_entry": [],
                "time_of_exit": [],
                "bid_increment": [],
                "autobid": []}
# %% Make sure to only keep values where the valid bid count is equal to the number of bids in the dataset
#Iterate over each lot_id
for lot_id in tqdm(df["lot.id"].unique()):
    df_this_lot = df[df["lot.id"] == lot_id]
    # Check if valid_bid_count is equal to the number of bids in the data for this lot 
    if df_this_lot["lot.valid_bid_count"].mean() == df_this_lot.shape[0]:
        # Iterate over each user that made a bid for this lot
        for user_id in df[df["lot.id"] == lot_id]["user.id"].unique():
            #For this lot, user combi, which has a bidding behaviour, create features
            df_user_lot = df[(df["lot.id"] == lot_id) & (df["user.id"] == user_id)]
            print(df_user_lot)
            # Store the lot-user combination
            cluster_dict["lot_user_combi"].append((lot_id, user_id))
            
            ## Find the number of bids by this user
            cluster_dict["nr_bids"].append(df_user_lot.shape[0])
            
            ## Find the time of entry
            # Sort values by bidding time in chronological order
            df_user_lot = df_user_lot.sort_values(
                by="bid.date").reset_index(drop=True)
            #Find the moment of the first bid in hours after lot startingmoment
            cluster_dict["time_of_entry"].append(
                (df_user_lot["bid.date"][0] - df_user_lot["lot.startingdate"][0]) / pd.Timedelta("1 hour"))

            ## Find the time of exit
            cluster_dict["time_of_exit"].append(
                (df_user_lot["bid.date"][df_user_lot.shape[0] - 1]
                 - df_user_lot["lot.startingdate"][df_user_lot.shape[0] - 1]) / pd.Timedelta("1 hour"))

            ## Find the average bid increment
            #TODO standardize the values
            df_this_lot_sorted = df_this_lot.sort_values(by="bid.date").reset_index(drop=True)
            for index in range(df_this_lot_sorted.shape[0]):
                if df_this_lot_sorted.iloc[index]["user.id"] == user_id:
                    if index == 0:
                        #Increment = bid amount - starting price
                        try:
                            increment = df_this_lot_sorted.iloc[index]["bid.amount"] - df_this_lot_sorted.iloc[index]["lot.start_amount"]
                        # If the starting price is 0 and this causes a division error, pretend starting price was 1
                        except ZeroDivisionError:
                            increment = df_this_lot_sorted.iloc[index]["bid.amount"] - 1
                    else:
                        increment = df_this_lot_sorted.iloc[index]["bid.amount"] - df_this_lot_sorted.iloc[index - 1]["bid.amount"]
                    #TODO take aggregate value of the increments    
            
            
            # if len(df_user_lot["bid.is_autobid"].unique()) > 1:
            #     print("Length more than 1")
            # print(df_user_lot[["lot.id", "lot.valid_bid_count"]])
# %%
# Nr. of bids
# Time of first bid
# Time of last bid
# Average increment per bid
# Autobid
# 
