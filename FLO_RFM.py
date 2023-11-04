###############################################################
# Customer Segmentation with RFM
###############################################################

###############################################################
# Understanding the Data
###############################################################
import pandas as pd
import datetime as dt

pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: "%.5f" % x)
pd.set_option("display.width", 100)

df_ = pd.read_csv("flo_data_20k.csv")
df = df_.copy()


def check_df(dataframe=pd.DataFrame):
    """
    First view of your Data to understand the concept of your observation units

    Parameters
    ----------
    dataframe : DataFrame

    Returns
    -------

    """
    print("##############  HEAD ###################")
    print(dataframe.head(10))
    print("##############  SHAPE  #################")
    print(dataframe.shape)
    print("###############  NA  ####################")
    print(dataframe.isnull().sum())
    print("##############  INFO  ###################")
    print(dataframe.info())
    print("#####################################")


check_df(df)


# Scripting the data preparation process.
def data_prep(dataframe, date_var=None):
    """
    Prepares and analyzes data from the given DataFrame.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The input DataFrame containing data to be processed.

    date_var : str, optional
        The name of the date variable in the DataFrame. If provided, the date variable will be converted to datetime format.

    Returns
    -------
    pd.DataFrame
        Summary statistics for order channels, including total orders and total customer value.

    pd.Series
        The top 10 customers with the highest total value spent.

    pd.Series
        The top 10 customers with the highest total number of orders.
    """
    # Creating new variables for total orders and total value spent
    dataframe["order_num_total_ever"] = (
        dataframe["order_num_total_ever_online"]
        + dataframe["order_num_total_ever_offline"]
    )

    dataframe["customer_value_total_ever"] = (
        dataframe["customer_value_total_ever_online"]
        + dataframe["customer_value_total_ever_offline"]
    )

    # Changing the type of variables containing dates to datetime
    if date_var is not None:
        dataframe[date_var] = dataframe[date_var].apply(pd.to_datetime)

    # Distribution of the number of customers in shopping channels, total number of products purchased and total expenditures.
    order_channel_summary = (
        df.groupby("order_channel")
        .agg({"order_num_total_ever": "sum", "customer_value_total_ever": "sum"})
        .describe()
        .T
    )

    # Top 10 customers with the most revenue
    sorted_df_by_total_value = df.sort_values(
        by="customer_value_total_ever", ascending=False
    )
    top_10_value_customer = sorted_df_by_total_value["master_id"].head(10)

    # Top 10 customers with the most orders
    sorted_df_by_total_order = df.sort_values(
        by="order_num_total_ever", ascending=False
    )
    top_10_order_customer = sorted_df_by_total_order["master_id"].head(10)

    return (
        order_channel_summary,
        top_10_value_customer,
        top_10_order_customer,
    )


date_variables = df.columns[df.columns.str.contains("date")]

data_prep(df, date_variables)
# df.info()
# df.head()

###############################################################
# Calculating RFM Metrics
###############################################################

# Analysis date 2 days after the date of the last purchase in the dataset
df["last_order_date"].sort_values(ascending=False).head()

# last_order_date == "2021-05-30"
today_date = pd.Timestamp(dt.date(2021, 6, 1))

# Creating an rfm dataframe with customer_id, recency, frequnecy and monetary.
rfm = df.groupby("master_id").agg(
    {
        "last_order_date": lambda date: (today_date - date).dt.days,
        "order_num_total_ever": "sum",
        "customer_value_total_ever": "sum",
    }
)

rfm.columns = ["recency", "frequency", "monetary"]

###############################################################
# Calculating RF and RFM Scores
###############################################################

# Convert Recency, Frequency and Monetary metrics into scores between 1-5 with the help of qcut and save these scores as recency_score, frequency_score and monetary_score
rfm["recency_score"] = pd.qcut(rfm["recency"], 5, labels=[5, 4, 3, 2, 1])
rfm["frequency_score"] = pd.qcut(
    rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]
)
rfm["monetary_score"] = pd.qcut(rfm["monetary"], 5, labels=[1, 2, 3, 4, 5])

# rfm.head()

# Expressing recency_score and frequency_score as a single variable and saving as RF_SCORE
rfm["RF_SCORE"] = rfm["recency_score"].astype(str) + rfm["frequency_score"].astype(str)

###############################################################
# Defining RF Scores as Segments
###############################################################

# Defining segments to make the generated RFM scores more readable and translating RF_SCORE into segments with the help of the defined seg_map
seg_map = {
    r"[1-2][1-2]": "hibernating",
    r"[1-2][3-4]": "at_Risk",
    r"[1-2]5": "cant_lose",
    r"3[1-2]": "about_to_sleep",
    r"33": "need_attention",
    r"[3-4][4-5]": "loyal_customers",
    r"41": "promising",
    r"51": "new_customers",
    r"[4-5][2-3]": "potential_loyalists",
    r"5[4-5]": "champions",
}
# rfm.head()
rfm["segment"] = rfm["RF_SCORE"].replace(seg_map, regex=True)

# Examining the recency, frequnecy and monetary averages of the segments.
rfm[["segment", "recency", "frequency", "monetary"]].groupby("segment").agg(
    ["mean", "count"]
)

# FLO includes a new women's shoe brand in its organization. The product prices of the brand are above the general customer preferences. For this reason, it is desired to contact the customers who will be interested in the promotion of the brand and product sales. These customers are planned to be loyal and female category shoppers. Save the id numbers of the customers in csv file as new_brand_target_customer_id.csv.

loyal_customers = rfm[
    (rfm["segment"] == "loyal_customers") | (rfm["segment"] == "champions")
].index
woman_cat = df[df["interested_in_categories_12"].str.contains("KADIN")]["master_id"]
loyal_woman_customers = [
    customer for customer in woman_cat if customer in loyal_customers
]
loyal_woman_customers = pd.DataFrame(
    {"loyal_woman_customer_ids": loyal_woman_customers}
)
loyal_woman_customers.to_csv("flo_woman.csv")  # index = False argümanını ekle.

# b. Erkek ve Çoçuk ürünlerinde %40'a yakın indirim planlanmaktadır. Bu indirimle ilgili kategorilerle ilgilenen geçmişte iyi müşterilerden olan ama uzun süredir
# alışveriş yapmayan ve yeni gelen müşteriler özel olarak hedef alınmak isteniliyor. Uygun profildeki müşterilerin id'lerini csv dosyasına indirim_hedef_müşteri_ids.csv
# olarak kaydediniz.
cant_lose_and_new_customers = rfm[
    (rfm["segment"] == "cant_lose")
    | (rfm["segment"] == "new_customers")
    | (rfm["segment"] == "hibernating")
].index

erkek_and_cocuk_cat = df[
    df["interested_in_categories_12"].str.contains("ERKEK|COCUK", case=False)
]["master_id"]

discount_target = [
    target for target in erkek_and_cocuk_cat if target in cant_lose_and_new_customers
]

discount_target = pd.DataFrame({"target_id": discount_target})
discount_target.to_csv("discount_target_customers_ids.csv")
