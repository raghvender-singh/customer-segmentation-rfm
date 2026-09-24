import pandas as pd
import numpy as np


# ==========================================
# 1. LOAD DATA
# ==========================================

df = pd.read_csv("data/online_retail_II.csv")

print(df.shape)
print(df.info())


# ==========================================
# 2. CHECK MISSING VALUES
# ==========================================

print(df.isnull().sum())


# ==========================================
# 3. REMOVE CUSTOMERS WITH MISSING CUSTOMER ID
# ==========================================

df = df.dropna(subset=["Customer ID"])

print(df.shape)
print(df["Customer ID"].isnull().sum())


# ==========================================
# 4. REMOVE DUPLICATE ROWS
# ==========================================

print("Duplicate rows:", df.duplicated().sum())

df = df.drop_duplicates(ignore_index=True)

print("Duplicate rows after cleaning:", df.duplicated().sum())
print(df.shape)


# ==========================================
# 5. CONVERT INVOICE DATE TO DATETIME
# ==========================================

df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

print(df["InvoiceDate"].dtype)


# ==========================================
# 6. REMOVE INVALID QUANTITY AND PRICE
# ==========================================

print("Quantity <= 0:", (df["Quantity"] <= 0).sum())
print("Price <= 0:", (df["Price"] <= 0).sum())

df = df[
    (df["Quantity"] > 0) &
    (df["Price"] > 0)
]

print(df.shape)

print("Quantity <= 0:", (df["Quantity"] <= 0).sum())
print("Price <= 0:", (df["Price"] <= 0).sum())


# ==========================================
# 7. CREATE REVENUE
# ==========================================

df["Revenue"] = df["Quantity"] * df["Price"]

print(df[["Quantity", "Price", "Revenue"]].head())


# ==========================================
# 8. CHECK DATE RANGE
# ==========================================

print("Minimum date:", df["InvoiceDate"].min())
print("Maximum date:", df["InvoiceDate"].max())


# ==========================================
# 9. SET REFERENCE DATE
# ==========================================

reference_date = pd.Timestamp("2011-12-10")


# ==========================================
# 10. RECENCY
# ==========================================

customer_last_purchase = (
    df.groupby("Customer ID")["InvoiceDate"]
    .max()
)

recency = reference_date - customer_last_purchase

recency = recency.dt.days


# ==========================================
# 11. FREQUENCY
# ==========================================

frequency = (
    df.groupby("Customer ID")["Invoice"]
    .nunique()
)


# ==========================================
# 12. MONETARY
# ==========================================

monetary = (
    df.groupby("Customer ID")["Revenue"]
    .sum()
)


# ==========================================
# 13. CREATE RFM TABLE
# ==========================================

rfm = pd.concat(
    [recency, frequency, monetary],
    axis=1
)

rfm.columns = [
    "Recency",
    "Frequency",
    "Monetary"
]

print(rfm.head())


# ==========================================
# 14. RFM SUMMARY
# ==========================================

print(rfm.describe())


# ==========================================
# 15. RECENCY SCORE
# ==========================================

rfm["R_Score"] = pd.qcut(
    rfm["Recency"],
    5,
    labels=[5, 4, 3, 2, 1]
)


# ==========================================
# 16. FREQUENCY SCORE
# ==========================================

rfm["F_Score"] = pd.qcut(
    rfm["Frequency"].rank(method="first"),
    5,
    labels=[1, 2, 3, 4, 5]
)


# ==========================================
# 17. MONETARY SCORE
# ==========================================

rfm["M_Score"] = pd.qcut(
    rfm["Monetary"].rank(method="first"),
    5,
    labels=[1, 2, 3, 4, 5]
)


# ==========================================
# 18. CONVERT SCORES TO INTEGER
# ==========================================

rfm["R_Score"] = rfm["R_Score"].astype(int)
rfm["F_Score"] = rfm["F_Score"].astype(int)
rfm["M_Score"] = rfm["M_Score"].astype(int)


# ==========================================
# 19. CREATE RFM SCORE CODE
# ==========================================

rfm["RFM_Score"] = (
    rfm["R_Score"].astype(str)
    + rfm["F_Score"].astype(str)
    + rfm["M_Score"].astype(str)
)


# ==========================================
# 20. CREATE TOTAL RFM SCORE
# ==========================================

rfm["RFM_Total"] = (
    rfm["R_Score"]
    + rfm["F_Score"]
    + rfm["M_Score"]
)


# ==========================================
# CHECK FINAL RFM TABLE
# ==========================================

print(
    rfm[
        [
            "Recency",
            "Frequency",
            "Monetary",
            "R_Score",
            "F_Score",
            "M_Score",
            "RFM_Score",
            "RFM_Total"
        ]
    ].head(10)
)


# -----------------------------
# CUSTOMER SEGMENTATION
# -----------------------------

segment_conditions = [
    # Champions
    (rfm["R_Score"] >= 4) &
    (rfm["F_Score"] >= 4) &
    (rfm["M_Score"] >= 4),

    # Loyal Customers
    (rfm["R_Score"] >= 3) &
    (rfm["F_Score"] >= 4) &
    (rfm["M_Score"] >= 3),

    # Potential Loyalists
    (rfm["R_Score"] >= 4) &
    (rfm["F_Score"] <= 3) &
    (rfm["M_Score"] >= 2),

    # At Risk
    (rfm["R_Score"] <= 2) &
    (rfm["F_Score"] >= 3) &
    (rfm["M_Score"] >= 3),

    # Hibernating
    (rfm["R_Score"] <= 2) &
    (rfm["F_Score"] <= 2) &
    (rfm["M_Score"] <= 2)
]

segment_labels = [
    "Champions",
    "Loyal Customers",
    "Potential Loyalists",
    "At Risk",
    "Hibernating"
]

rfm["Segment"] = np.select(
    segment_conditions,
    segment_labels,
    default="Others"
)

print("\nCUSTOMER SEGMENTS:")
print(rfm["Segment"].value_counts())


# -----------------------------
# SEGMENT-WISE RFM SUMMARY
# -----------------------------

segment_summary = (
    rfm.groupby("Segment")[["Recency", "Frequency", "Monetary"]]
    .mean()
)

print("\nSEGMENT SUMMARY:")
print(segment_summary)


# -----------------------------
# CUSTOMER PERCENTAGE BY SEGMENT
# -----------------------------

segment_percentage = (
    rfm["Segment"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\nCUSTOMER PERCENTAGE:")
print(segment_percentage)


# -----------------------------
# REVENUE BY SEGMENT
# -----------------------------

segment_revenue = (
    rfm.groupby("Segment")["Monetary"]
    .sum()
    .sort_values(ascending=False)
)

print("\nREVENUE BY SEGMENT:")
print(segment_revenue)


# -----------------------------
# REVENUE PERCENTAGE BY SEGMENT
# -----------------------------

segment_revenue_percentage = (
    segment_revenue
    / segment_revenue.sum()
    * 100
).round(2)

print("\nREVENUE PERCENTAGE:")
print(segment_revenue_percentage)


# -----------------------------
# VISUALIZATION 1: CUSTOMER COUNT
# -----------------------------

import matplotlib.pyplot as plt

segment_counts = rfm["Segment"].value_counts()

segment_counts.plot(kind="bar")

plt.title("Customer Count by Segment")
plt.xlabel("Customer Segment")
plt.ylabel("Number of Customers")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# -----------------------------
# VISUALIZATION 2: REVENUE BY SEGMENT
# -----------------------------

segment_revenue.plot(kind="bar")

plt.title("Revenue by Customer Segment")
plt.xlabel("Customer Segment")
plt.ylabel("Total Revenue")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# -----------------------------
# EXPORT FINAL RFM DATASET
# -----------------------------

rfm.to_csv(
    "data/rfm_customer_segments.csv",
    index=True
)

print("\nFINAL RFM DATASET EXPORTED SUCCESSFULLY.")