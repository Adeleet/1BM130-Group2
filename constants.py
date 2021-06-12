COLNAMES_DIM_AUCTION = {
    "auction_id": "auction.id",
    "auction_name": "auction.name",
    "auction_country": "auction.country",
    "country_isocode": "auction.country_isocode",
    "startdate": "auction.start_date",
    "closedate": "auction.close_date",
    "onlinedate": "auction.online_date",
    "is_private": "auction.is_private",
    "is_homedelivery": "auction.is_homedelivery",
    "is_public": "auction.is_public",
    "open_for_bidding": "auction.is_open_for_bidding",
    "is_closed": "auction.is_closed",
    "sourcing_company": "auction.sourcing_company",
}


COLNAMES_AUCTION_CLOSE_TIMES = {
    "AUCTIONID": "auction.id",
    "AUCTIONSTARTDATE": "auction.start_date",
    "AUCTIONENDDATE": "auction.end_date",
    "LOTMINSTARTDATE": "auction.lot_min_start_date",
    "LOTMAXSTARTDATE": "auction.lot_max_start_date",
    "LOTMINENDDATE": "auction.lot_min_end_date",
    "LOTMAXENDDATE": "auction.lot_max_end_date",
    "BIDRANGENAME": "auction.bidrange_name",
}

COLNAMES_LOTS = {
    "lot_id": "lot.id",
    "auction_id": "auction.id",
    "lot_number": "lot.number",
    "auction_lotnumber": "lot.auction_lot_number",
    "seller_id_an": "lot.seller_id",
    "opportunity_id": "lot.opportunity_id",
    "naam": "lot.title",
    "is_cancelled": "lot.is_cancelled",
    "closingdate": "lot.closingdate_day",
    "startingdate": "lot.startingdate",
    "newprice": "lot.newprice",
    "is_open": "lot.is_open",
    "auctionfeetaxrate": "lot.taxrate",
    "is_uitgeleverd": "lot.is_collected",
    "uitleverdatum": "lot.collection_date",
    "lot_subcategory": "lot.subcategory",
    "lot_topcategory": "lot.category",
    "has_bid": "lot.has_bid",
    "valid_bid_count": "lot.valid_bid_count",
    "is_invoiced": "lot.is_invoiced",
    "is_1euro_kavel": "lot.starting_at_1EUR",
    "is_sold": "lot.is_sold",
    "is_credited": "lot.is_credited",
    "latest_creditreason": "lot.credit_reason",
    "brand": "lot.brand",
    "condition": "lot.condition",
    "type": "lot.type",
    "categorycode": "lot.category_code",
    "category_seller": "lot.seller_category",
}

COLNAMES_PROJECTS = {
    "auction_id": "auction.id",
    "project_accountmanager": "project.accountmanager",
    "project_businessline": "project.business_line",
    "project_businessunit": "project.business_unit",
    "main_category": "auction.main_category",
    "is_homedelivery": "project.is_homedelivery",
    "is_public": "project.is_public",
    "is_automatic_credit": "project.is_automatic_credit",
}

COLNAMES_BIDS = {
    "bid_id": "bid.id",
    "user_id": "user.id",
    "lot_id": "lot.id",
    "auction_id": "auction.id",
    "is_autobid": "bid.is_autobid",
    "lot_closingdate": "lot.closingdate",
    "auction_closingdate": "auction.closingdate",
    "is_valid": "bid.is_valid",
    "bid_amount": "bid.amount",
    "latest_bid": "bid.is_latest",
    "first_bid": "bid.is_first",
    "days_to_close": "bid.days_to_close",
    "bid_date": "bid.date",
    "added_bidvalue": "bid.added_bidvalue",
    "efficy_business_line": "bid.efficy_business_line",
}

COLNAMES_FACT_LOTS = {
    "Sum of app_lotviews": "lot.app_views",
    "auction_id": "auction.id",
    "categorycode": "lot.category_code",
    "closingdate": "lot.closingdate",
    "lot_id": "lot.id",
    "lot_newprice": "lot.new_price",
    "opportunity_id": "lot.opportunity_id",
    "startamount": "lot.start_amount",
    "startingdate": "lot.startingdate",
    "Sum of web_lotviews": "lot.web_views",
}

TRAIN_COLS_IS_SOLD = [
    "auction.is_homedelivery",
    "auction.sourcing_company",
    "auction.bidrange_name",
    "auction.is_public",
    "auction.num_lots",
    "lot.taxrate",
    "lot.closing_count",
    "lot.category_closing_count",
    "lot.subcategory_closing_count",
    "lot.timeslot_closing_count",
    "lot.timeslot_category_closing_count",
    "lot.timeslot_subcategory_closing_count",
    "lot.category",
    "lot.days_open",
    "project.business_line",
    "project.business_unit",
    "project.is_homedelivery",
    "project.is_public",
    "project.accountmanager",
    "lot.starting_at_1EUR",
    "lot.start_amount",
    "lot.is_sold",
    "lot.closing_day_of_week",
    "lot.category_count_in_auction",
    "lot.subcategory_count_in_auction",
    "lot.rel_nr",
]
TRAIN_COLS_REVENUE = [
    "auction.is_homedelivery",
    "auction.sourcing_company",
    "auction.bidrange_name",
    "auction.num_lots",
    "auction.is_public",
    "lot.taxrate",
    "lot.closing_count",
    "lot.category_closing_count",
    "lot.subcategory_closing_count",
    "lot.timeslot_closing_count",
    "lot.timeslot_category_closing_count",
    "lot.timeslot_subcategory_closing_count",
    "lot.category",
    "lot.days_open",
    "project.business_line",
    "project.business_unit",
    "project.is_homedelivery",
    "project.is_public",
    "project.accountmanager",
    "lot.starting_at_1EUR",
    "lot.start_amount",
    "lot.revenue",
    "lot.closing_day_of_week",
    "lot.category_count_in_auction",
    "lot.subcategory_count_in_auction",
    "lot.rel_nr",
]
