# %%
import pandas as pd
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
         "lot.closingdate", "lot.startingdate","lot.has_bid",
         "lot.valid_bid_count", "lot.starting_at_1EUR", "lot.is_sold", "bid.id",
         "user.id", "bid.is_autobid", "bid.is_valid", "bid.amount",
         "bid.is_latest", "bid.is_first", "bid.days_to_close",
         "bid.added_bidvalue", "lot.start_amount"]]

# %%
bidding_df = pd.DataFrame(df["user.id"])
bidding_df["lot.id"] = df["lot.id"]
# Only keep unique combinations of user id and lot id
bidding_df.drop_duplicates(inplace = True)
# %%
