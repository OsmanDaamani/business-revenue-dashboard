# predict.py  (Streamlit dashboard - final, single-file)
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Business Dashboard (Pizza Demo)", layout="wide")
st.title("Business Dashboard — Sales, Revenue & Forecast (Demo)")

# -------------------- Load ML Artifacts --------------------
rev_model = joblib.load("pizza_revenue_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_cols = joblib.load("feature_columns.pkl")

# -------------------- Data Loader --------------------
st.sidebar.header("1) Load Data")
uploaded = st.sidebar.file_uploader("Upload merged dataset (CSV)", type=["csv"])

@st.cache_data(show_spinner=False)
def load_df_from_csv(file) -> pd.DataFrame:
    df0 = pd.read_csv(file)
    return df0

if uploaded is None:
    st.info("Upload your merged CSV (with columns like: order_id, date, time, name, size, category, quantity, price).")
    st.stop()

df = load_df_from_csv(uploaded).copy()

# -------------------- Required Columns --------------------
required = {"order_id", "date", "time", "name", "size", "category", "quantity", "price"}
missing = required - set(df.columns)
if missing:
    st.error(f"Missing required columns: {sorted(list(missing))}")
    st.stop()

# -------------------- Cleaning / Feature Prep --------------------
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["time"] = pd.to_datetime(df["time"], format="%H:%M:%S", errors="coerce").dt.time
df["hour"] = pd.to_datetime(df["time"].astype(str), format="%H:%M:%S", errors="coerce").dt.hour
df["weekday"] = df["date"].dt.weekday
df["revenue"] = df["price"] * df["quantity"]

# Fallback for weird parses
df["hour"] = df["hour"].fillna(0).astype(int)
df["weekday"] = df["weekday"].fillna(0).astype(int)

# -------------------- Filters --------------------
st.sidebar.header("2) Filters")
min_d = df["date"].min()
max_d = df["date"].max()
date_range = st.sidebar.date_input("Date range", value=(min_d.date(), max_d.date()))
if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    d1, d2 = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
else:
    d1, d2 = min_d, max_d

sizes = ["All"] + sorted(df["size"].dropna().astype(str).unique().tolist())
cats = ["All"] + sorted(df["category"].dropna().astype(str).unique().tolist())

f_size = st.sidebar.selectbox("Size", sizes, index=0)
f_cat = st.sidebar.selectbox("Category", cats, index=0)
top_n = st.sidebar.slider("Top N items", 5, 50, 20)

mask = (df["date"] >= d1) & (df["date"] <= d2)
if f_size != "All":
    mask &= (df["size"].astype(str) == f_size)
if f_cat != "All":
    mask &= (df["category"].astype(str) == f_cat)

fdf = df.loc[mask].copy()
if fdf.empty:
    st.warning("No data after filters. Adjust filters.")
    st.stop()

# -------------------- KPIs --------------------
total_revenue = float(fdf["revenue"].sum())
total_qty = int(fdf["quantity"].sum())
total_orders = int(fdf["order_id"].nunique())
avg_items_per_order = float(fdf.groupby("order_id")["quantity"].sum().mean())
avg_rev_per_order = float(fdf.groupby("order_id")["revenue"].sum().mean())

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Revenue", f"{total_revenue:,.2f}")
k2.metric("Total Quantity", f"{total_qty:,}")
k3.metric("Total Orders", f"{total_orders:,}")
k4.metric("Avg Items / Order", f"{avg_items_per_order:,.2f}")
k5.metric("Avg Revenue / Order", f"{avg_rev_per_order:,.2f}")

# -------------------- Charts --------------------
left, right = st.columns(2)

# Revenue by Date
rev_by_date = fdf.groupby(fdf["date"].dt.date)["revenue"].sum().reset_index()
rev_by_date.columns = ["date", "revenue"]
fig1 = px.line(rev_by_date, x="date", y="revenue", markers=True, title="Revenue by Date")
left.plotly_chart(fig1, use_container_width=True)

# Revenue by Hour
rev_by_hour = fdf.groupby("hour")["revenue"].sum().reset_index()
fig2 = px.bar(rev_by_hour, x="hour", y="revenue", title="Revenue by Hour")
right.plotly_chart(fig2, use_container_width=True)

left2, right2 = st.columns(2)

# Top Items by Revenue
top_items = (
    fdf.groupby(["name", "size"])["revenue"]
    .sum()
    .sort_values(ascending=False)
    .head(top_n)
    .reset_index()
)
top_items["item"] = top_items["name"].astype(str) + " (" + top_items["size"].astype(str) + ")"
fig3 = px.bar(top_items[::-1], x="revenue", y="item", orientation="h", title=f"Top {top_n} Items by Revenue")
left2.plotly_chart(fig3, use_container_width=True)

# Revenue by Size
rev_by_size = fdf.groupby("size")["revenue"].sum().sort_values(ascending=False).reset_index()
fig4 = px.bar(rev_by_size, x="size", y="revenue", title="Revenue by Size")
right2.plotly_chart(fig4, use_container_width=True)

left3, right3 = st.columns(2)

# Quantity by Category
qty_by_cat = fdf.groupby("category")["quantity"].sum().sort_values(ascending=False).reset_index()
fig5 = px.bar(qty_by_cat, x="category", y="quantity", title="Quantity by Category")
left3.plotly_chart(fig5, use_container_width=True)

# Weekend vs Weekday (Revenue)
fdf["week_type"] = np.where(fdf["weekday"].isin([4, 5]), "Weekend", "Weekday")
rev_week = fdf.groupby("week_type")["revenue"].sum().reset_index()
fig6 = px.bar(rev_week, x="week_type", y="revenue", title="Revenue: Weekend vs Weekday")
right3.plotly_chart(fig6, use_container_width=True)

# -------------------- Recommendations (simple, actionable) --------------------
st.subheader("Recommendations (Actionable Summary)")

peak_hour = int(rev_by_hour.sort_values("revenue", ascending=False).iloc[0]["hour"])
best_size = str(rev_by_size.sort_values("revenue", ascending=False).iloc[0]["size"])
best_cat = str(qty_by_cat.sort_values("quantity", ascending=False).iloc[0]["category"])
top_item_name = top_items.iloc[0]["item"] if not top_items.empty else "N/A"

rec1, rec2, rec3 = st.columns(3)
rec1.info(f"**Peak time:** Focus staffing & upsells around **{peak_hour}:00** (highest hourly revenue).")
rec2.info(f"**Menu emphasis:** **{best_cat}** leads volume; highlight it in offers and menu positioning.")
rec3.info(f"**Best driver:** Size **{best_size}** is strongest for revenue; build bundles around it.\n\n**Top item:** {top_item_name}")

# -------------------- Forecast (Model) + Scenario Comparison --------------------
st.subheader("Forecast (Model) — Not a Calculator")

cA, cB, cC, cD, cE = st.columns(5)
with cA:
    in_quantity = st.number_input("Quantity", min_value=1, value=2, step=1)
with cB:
    in_hour = st.slider("Hour", 0, 23, int(peak_hour))
with cC:
    in_weektype = st.selectbox("Week Type", [("Weekday", 2), ("Weekend", 5)], format_func=lambda x: x[0])[1]
with cD:
    in_size = st.selectbox("Size (Forecast)", ["S", "M", "L", "XL", "XXL"], index=min(1, 4))
with cE:
    in_category = st.selectbox("Category (Forecast)", ["Classic", "Supreme", "Veggie", "Chicken"], index=0)

def predict_revenue(quantity, hour, weekday, size, category) -> float:
    sample = pd.DataFrame([{
        "quantity": quantity,
        "hour": hour,
        "weekday": weekday,
        "size": size,
        "category": category
    }])
    sample = pd.get_dummies(sample)
    sample = sample.reindex(columns=feature_cols, fill_value=0)
    sample_scaled = scaler.transform(sample)
    return float(rev_model.predict(sample_scaled)[0])

p1, p2, p3 = st.columns(3)

base_pred = predict_revenue(in_quantity, in_hour, in_weektype, in_size, in_category)
p1.metric("Predicted Revenue", f"{base_pred:,.2f}")

# scenario 1: +3 hours
alt_hour = min(in_hour + 3, 23)
pred_hour = predict_revenue(in_quantity, alt_hour, in_weektype, in_size, in_category)
p2.metric("Scenario: +3 hours", f"{pred_hour:,.2f}", delta=f"{(pred_hour - base_pred):,.2f}")

# scenario 2: switch week type
swap_week = 5 if in_weektype == 2 else 2
pred_week = predict_revenue(in_quantity, in_hour, swap_week, in_size, in_category)
p3.metric("Scenario: Switch Week Type", f"{pred_week:,.2f}", delta=f"{(pred_week - base_pred):,.2f}")

st.caption(
    "Why this is not a calculator: the model learned patterns from historical behavior (time/day/size/category effects), "
    "so the same quantity can produce different expected revenue under different conditions."
)

# -------------------- Download --------------------
st.subheader("Export (Filtered Data)")
csv = fdf.to_csv(index=False).encode("utf-8")
st.download_button("Download filtered CSV", data=csv, file_name="filtered_business_data.csv", mime="text/csv")
