# =============================================================================
# PHARMACY CHAIN ANALYTICS — STREAMLIT APP
# =============================================================================
# Run with:  streamlit run pharmacy_streamlit_app.py
#
# Install dependencies first:
#   pip install streamlit pandas numpy matplotlib seaborn scikit-learn
#              mlxtend tensorflow openpyxl
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="Pharmacy Chain Analytics",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("💊 Pharmacy Chain Analytics & Predictive Modeling")
st.markdown("Upload your pharmacy Excel file to run the full analytics pipeline.")

# =============================================================================
# SIDEBAR — FILE UPLOAD & OPTIONS
# =============================================================================
st.sidebar.header("⚙️ Settings")
uploaded_file = st.sidebar.file_uploader(
    "Upload Excel file", type=["xlsx"],
    help="pharmacy_chain_curation_project_with_prescriptions.xlsx"
)

run_ann = st.sidebar.checkbox("Run ANN Demand Forecasting (slow)", value=False)
ann_epochs = st.sidebar.slider("ANN Epochs", 10, 100, 50, step=10)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Parts covered:**\n"
    "1. Data Integration & Preprocessing\n"
    "2. Association Rule Mining\n"
    "3. Classification (LR, DT, RF, SVM)\n"
    "4. Customer Clustering\n"
    "5. ANN Demand Forecasting\n"
    "6. Advanced Analytics\n"
    "7. Visualizations"
)

if uploaded_file is None:
    st.warning("👈 Please upload your Excel file in the sidebar to begin.")
    st.stop()

# =============================================================================
# CACHING — load & preprocess data once
# =============================================================================
@st.cache_data(show_spinner="Loading and preprocessing data…")
def load_and_preprocess(file_bytes):
    xl = pd.ExcelFile(file_bytes)

    def clean_cols(df):
        df.columns = df.columns.str.strip().str.lower().str.replace(r"\s+", "_", regex=True)
        return df

    customers     = clean_cols(xl.parse("Customers"))
    medicines     = clean_cols(xl.parse("Medicines"))
    med_types     = clean_cols(xl.parse("TypesOfMedicines"))
    shops         = clean_cols(xl.parse("PharmacyShops"))
    prescriptions = clean_cols(xl.parse("Prescriptions"))
    sales         = clean_cols(xl.parse("SalesBills"))
    purchases     = clean_cols(xl.parse("Purchases"))
    stocks        = clean_cols(xl.parse("Stocks"))

    # Handle missing values
    for df in [customers, medicines, sales, prescriptions]:
        for col in df.columns:
            if df[col].isnull().any():
                if df[col].dtype in [np.float64, np.int64]:
                    df[col].fillna(df[col].median(), inplace=True)
                else:
                    df[col].fillna(df[col].mode().iloc[0], inplace=True)

    prescriptions.drop_duplicates(inplace=True)
    medicines["medicine_name"] = medicines["medicine_name"].str.strip().str.title()

    # Detect category & stock columns
    possible_cat_cols = [c for c in med_types.columns if c not in ["type_id"]]
    CAT_COL = possible_cat_cols[0] if possible_cat_cols else None
    if CAT_COL and CAT_COL != "category":
        med_types = med_types.rename(columns={CAT_COL: "category"})

    stock_avail_candidates = [c for c in stocks.columns
                              if any(kw in c for kw in ["avail", "stock", "unit", "qty", "quantity"])]
    STOCK_COL = stock_avail_candidates[0] if stock_avail_candidates else stocks.columns[-1]
    if STOCK_COL != "available_units":
        stocks = stocks.rename(columns={STOCK_COL: "available_units"})

    medicines_full = medicines.merge(med_types, on="type_id", how="left")
    if "category" not in medicines_full.columns:
        medicines_full["category"] = medicines_full["medicine_name"]

    master = (
        sales
        .merge(customers,      on="customer_id", how="left")
        .merge(medicines_full, on="medicine_id", how="left")
        .merge(shops,          on="shop_id",     how="left")
    )

    pres_cols = ["customer_id", "medicine_id"]
    extra_pres_cols = [c for c in ["prescription_id", "doctor_name", "dosage"]
                       if c in prescriptions.columns]
    pres_lookup = prescriptions[pres_cols + extra_pres_cols].drop_duplicates(
        subset=["customer_id", "medicine_id"]
    )
    master = master.merge(pres_lookup, on=["customer_id", "medicine_id"], how="left")
    master = master.loc[:, ~master.columns.duplicated()]

    master["total_bill"] = master["quantity"] * master["price"]

    # Inject 30% OTC
    otc_idx = master.sample(frac=0.30, random_state=42).index
    master.loc[otc_idx, "prescription_id"] = np.nan
    master["has_prescription"] = master["prescription_id"].notna().astype(int)
    master["purchase_type"]    = master["has_prescription"].map({1: "Prescription", 0: "OTC"})
    master["sale_date"]        = pd.to_datetime(master["sale_date"], errors="coerce")
    master["sale_month"]       = master["sale_date"].dt.month
    master["sale_year"]        = master["sale_date"].dt.year

    cust_freq = master.groupby("customer_id").size().rename("cust_purchase_freq")
    master = master.merge(cust_freq, on="customer_id", how="left")

    med_freq = master.groupby("medicine_id").size().rename("med_demand_freq")
    master = master.merge(med_freq, on="medicine_id", how="left")

    otc_ratio = master.groupby("customer_id")["has_prescription"].mean().rename("prescription_ratio")
    master = master.merge(otc_ratio, on="customer_id", how="left")

    master = master.loc[:, ~master.columns.duplicated()]

    if isinstance(master["category"], pd.DataFrame):
        master["category"] = master["category"].iloc[:, 0]
    master["category"] = master["category"].astype(str)

    return master, stocks


master, stocks = load_and_preprocess(uploaded_file)

# =============================================================================
# TABS
# =============================================================================
tabs = st.tabs([
    "📋 Data Overview",
    "🔗 Association Rules",
    "🤖 Classification",
    "📊 Clustering",
    "🧠 ANN Forecasting",
    "📈 Advanced Analytics",
    "📉 Visualizations",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — DATA OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    st.header("📋 Part 1 — Data Integration & Preprocessing")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records",   f"{len(master):,}")
    col2.metric("Unique Customers", f"{master['customer_id'].nunique():,}")
    col3.metric("Unique Medicines", f"{master['medicine_id'].nunique():,}")
    col4.metric("Unique Shops",     f"{master['shop_id'].nunique():,}")

    st.subheader("Master DataFrame (first 10 rows)")
    st.dataframe(master.head(10), use_container_width=True)

    st.subheader("Column Names")
    st.write(list(master.columns))

    st.subheader("Missing Values")
    missing = master.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        st.success("No missing values in master DataFrame ✅")
    else:
        st.dataframe(missing.rename("missing_count"))

    st.subheader("Purchase Type Distribution")
    pt = master["purchase_type"].value_counts()
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.pie(pt, labels=pt.index, autopct="%1.1f%%", startangle=90)
    ax.set_title("Prescription vs OTC")
    st.pyplot(fig)
    plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — ASSOCIATION RULE MINING
# ─────────────────────────────────────────────────────────────────────────────
with tabs[1]:
    st.header("🔗 Part 2 — Association Rule Mining")

    basket = (
        master.groupby(["sale_id", "category"])["quantity"]
        .sum()
        .unstack(fill_value=0)
    )
    basket = basket.map(lambda x: 1 if x > 0 else 0).astype(bool)
    st.write(f"Basket shape: **{basket.shape[0]:,} transactions × {basket.shape[1]} categories**")

    n_transactions = len(basket)
    min_sup = max(0.01, 10 / n_transactions)
    st.info(f"Using min_support = **{min_sup:.4f}**")

    algo = st.radio("Choose Algorithm", ["Apriori", "FP-Growth"], horizontal=True)
    min_lift = st.slider("Minimum Lift", 0.5, 5.0, 1.0, 0.1)

    if st.button("Run Association Rule Mining"):
        with st.spinner("Mining rules…"):
            try:
                if algo == "Apriori":
                    freq = apriori(basket, min_support=min_sup, use_colnames=True)
                else:
                    freq = fpgrowth(basket, min_support=min_sup, use_colnames=True)

                if len(freq) > 1:
                    rules = association_rules(
                        freq, metric="lift", min_threshold=min_lift,
                        num_itemsets=len(freq)
                    )
                    st.success(f"✅ {len(rules)} rules found!")
                    st.subheader("Top Rules by Lift")
                    st.dataframe(
                        rules.sort_values("lift", ascending=False)
                             .head(20)
                             .reset_index(drop=True),
                        use_container_width=True
                    )
                    st.subheader("Top Frequent Itemsets")
                    st.dataframe(
                        freq.sort_values("support", ascending=False).head(10),
                        use_container_width=True
                    )
                else:
                    st.warning("Not enough frequent itemsets — try lowering min_support.")
            except Exception as e:
                st.error(f"Error: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────
with tabs[2]:
    st.header("🤖 Part 3 — Classification: Predict Purchase Type")

    le = LabelEncoder()
    location_candidates = [c for c in master.columns
                           if "location" in c or "city" in c or "area" in c]
    LOC_COL = location_candidates[0] if location_candidates else None

    clf_cols = ["age", "category", "discount",
                "cust_purchase_freq", "med_demand_freq",
                "prescription_ratio", "has_prescription"]
    if LOC_COL:
        clf_cols.insert(2, LOC_COL)

    clf_df = master[clf_cols].copy().dropna()
    if isinstance(clf_df["category"], pd.DataFrame):
        clf_df["category"] = clf_df["category"].iloc[:, 0]
    clf_df["category"]     = clf_df["category"].astype(str)
    clf_df["category_enc"] = le.fit_transform(clf_df["category"])

    feature_cols = ["age", "category_enc", "discount",
                    "cust_purchase_freq", "med_demand_freq", "prescription_ratio"]
    if LOC_COL:
        clf_df["location_enc"] = le.fit_transform(clf_df[LOC_COL].astype(str))
        feature_cols.insert(2, "location_enc")

    X = clf_df[feature_cols]
    y = clf_df["has_prescription"]

    st.subheader("Class Distribution")
    st.bar_chart(y.value_counts().rename({0: "OTC", 1: "Prescription"}))

    models_to_run = st.multiselect(
        "Select Models to Train",
        ["Logistic Regression", "Decision Tree", "Random Forest", "SVM"],
        default=["Logistic Regression", "Decision Tree", "Random Forest", "SVM"]
    )

    if st.button("Train Classifiers"):
        if y.nunique() < 2:
            st.error("Only one class present — cannot train classifier.")
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            scaler = StandardScaler()
            X_train_sc = scaler.fit_transform(X_train)
            X_test_sc  = scaler.transform(X_test)

            model_map = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Decision Tree":       DecisionTreeClassifier(random_state=42),
                "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
                "SVM":                 SVC(kernel="rbf", probability=True),
            }

            results = {}
            for name in models_to_run:
                model = model_map[name]
                with st.spinner(f"Training {name}…"):
                    model.fit(X_train_sc, y_train)
                    preds = model.predict(X_test_sc)
                    results[name] = {
                        "Accuracy":  round(accuracy_score(y_test, preds), 4),
                        "Precision": round(precision_score(y_test, preds, zero_division=0), 4),
                        "Recall":    round(recall_score(y_test, preds, zero_division=0), 4),
                        "F1":        round(f1_score(y_test, preds, zero_division=0), 4),
                    }

            results_df = pd.DataFrame(results).T
            st.subheader("📊 Model Comparison")
            st.dataframe(results_df, use_container_width=True)

            best = results_df["Accuracy"].idxmax()
            st.success(f"🏆 Best Model: **{best}** ({results_df.loc[best,'Accuracy']:.2%} accuracy)")

            # Confusion matrix for best model
            best_model = model_map[best]
            best_model.fit(X_train_sc, y_train)
            cm = confusion_matrix(y_test, best_model.predict(X_test_sc))
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                        xticklabels=["OTC", "Prescription"],
                        yticklabels=["OTC", "Prescription"])
            ax.set_title(f"Confusion Matrix — {best}")
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
            st.pyplot(fig)
            plt.close()

            # Bar chart
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            results_df[["Accuracy", "F1"]].plot(kind="bar", ax=ax2,
                                                  colormap="Set2", edgecolor="black")
            ax2.set_ylim(0, 1.1)
            ax2.set_title("Model Comparison: Accuracy vs F1")
            plt.xticks(rotation=20)
            st.pyplot(fig2)
            plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — CLUSTERING
# ─────────────────────────────────────────────────────────────────────────────
with tabs[3]:
    st.header("📊 Part 4 — Customer Segmentation (Clustering)")

    cust_profile = (
        master.groupby("customer_id")
        .agg(
            total_spending   = ("total_bill",       "sum"),
            purchase_freq    = ("sale_id",          "count"),
            pres_ratio       = ("has_prescription", "mean"),
            avg_discount     = ("discount",         "mean"),
            unique_medicines = ("medicine_id",      "nunique"),
        )
        .reset_index()
    )
    clust_features = ["total_spending", "purchase_freq",
                      "pres_ratio", "avg_discount", "unique_medicines"]
    cust_profile_clean = cust_profile[clust_features].dropna()
    cust_profile = cust_profile.loc[cust_profile_clean.index].reset_index(drop=True)

    scaler2 = MinMaxScaler()
    X_clust = scaler2.fit_transform(cust_profile[clust_features])

    # Elbow
    st.subheader("Elbow Method")
    inertias = []
    K_range = range(2, 11)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_clust)
        inertias.append(km.inertia_)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(list(K_range), inertias, "bo-")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Inertia")
    ax.set_title("Elbow Method for Optimal k")
    st.pyplot(fig)
    plt.close()

    OPTIMAL_K = st.slider("Choose number of clusters (k)", 2, 10, 4)

    if st.button("Run K-Means Clustering"):
        km_final = KMeans(n_clusters=OPTIMAL_K, random_state=42, n_init=10)
        cust_profile["kmeans_cluster"] = km_final.fit_predict(X_clust)

        st.subheader("Cluster Sizes")
        st.bar_chart(cust_profile["kmeans_cluster"].value_counts().sort_index())

        st.subheader("Cluster Profiles (Mean Values)")
        cluster_summary = cust_profile.groupby("kmeans_cluster")[clust_features].mean()
        st.dataframe(cluster_summary.style.background_gradient(cmap="Blues"),
                     use_container_width=True)

        fig2, ax2 = plt.subplots(figsize=(7, 5))
        scatter = ax2.scatter(
            cust_profile["total_spending"],
            cust_profile["purchase_freq"],
            c=cust_profile["kmeans_cluster"],
            cmap="viridis", alpha=0.6
        )
        plt.colorbar(scatter, ax=ax2, label="Cluster")
        ax2.set_xlabel("Total Spending")
        ax2.set_ylabel("Purchase Frequency")
        ax2.set_title("Customer Clusters (K-Means)")
        st.pyplot(fig2)
        plt.close()

        # Dendrogram
        st.subheader("Hierarchical Clustering Dendrogram")
        n_sample = min(200, len(X_clust))
        sample_idx = np.random.choice(len(X_clust), n_sample, replace=False)
        Z = linkage(X_clust[sample_idx], method="ward")
        fig3, ax3 = plt.subplots(figsize=(12, 5))
        dendrogram(Z, truncate_mode="lastp", p=20, ax=ax3)
        ax3.set_title(f"Dendrogram (sample={n_sample})")
        st.pyplot(fig3)
        plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — ANN FORECASTING
# ─────────────────────────────────────────────────────────────────────────────
with tabs[4]:
    st.header("🧠 Part 5 — ANN Demand Forecasting")

    if not run_ann:
        st.info("Enable **'Run ANN Demand Forecasting'** in the sidebar to activate this section.")
    else:
        demand = (
            master.groupby(["medicine_id", "sale_year", "sale_month"])
            .agg(
                demand       = ("quantity",         "sum"),
                avg_price    = ("price",            "mean"),
                avg_discount = ("discount",         "mean"),
                pres_ratio   = ("has_prescription", "mean"),
                stock_units  = ("med_demand_freq",  "first"),
            )
            .reset_index()
        )
        demand.dropna(inplace=True)

        ann_features = ["sale_year", "sale_month", "avg_price",
                        "avg_discount", "pres_ratio", "stock_units"]
        X_ann = demand[ann_features].values.astype(np.float32)
        y_ann = demand["demand"].values.astype(np.float32)

        st.write(f"Demand dataset: **{len(X_ann):,} samples**")

        if len(X_ann) < 10:
            st.error("Too few samples for ANN training.")
        elif st.button("Train ANN"):
            X_ann_train, X_ann_test, y_ann_train, y_ann_test = train_test_split(
                X_ann, y_ann, test_size=0.2, random_state=42
            )
            scaler3 = MinMaxScaler()
            X_ann_train = scaler3.fit_transform(X_ann_train)
            X_ann_test  = scaler3.transform(X_ann_test)

            tf.random.set_seed(42)
            ann = Sequential([
                Dense(64, activation="relu", input_shape=(X_ann_train.shape[1],)),
                Dropout(0.2),
                Dense(32, activation="relu"),
                Dropout(0.2),
                Dense(16, activation="relu"),
                Dense(1,  activation="linear"),
            ])
            ann.compile(optimizer="adam", loss="mse", metrics=["mae"])

            progress_bar = st.progress(0)
            status_text  = st.empty()

            history_log = {"loss": [], "val_loss": []}

            class StreamlitCallback(tf.keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    logs = logs or {}
                    history_log["loss"].append(logs.get("loss", 0))
                    history_log["val_loss"].append(logs.get("val_loss", 0))
                    progress_bar.progress(int((epoch + 1) / ann_epochs * 100))
                    status_text.text(
                        f"Epoch {epoch+1}/{ann_epochs} — "
                        f"loss: {logs.get('loss',0):.4f} | "
                        f"val_loss: {logs.get('val_loss',0):.4f}"
                    )

            ann.fit(
                X_ann_train, y_ann_train,
                validation_split=0.15,
                epochs=ann_epochs,
                batch_size=32,
                verbose=0,
                callbacks=[StreamlitCallback()]
            )

            loss, mae = ann.evaluate(X_ann_test, y_ann_test, verbose=0)
            y_pred = ann.predict(X_ann_test).flatten()
            rmse = float(np.sqrt(np.mean((y_ann_test - y_pred) ** 2)))

            col1, col2, col3 = st.columns(3)
            col1.metric("MAE",  f"{mae:.4f}")
            col2.metric("MSE",  f"{loss:.4f}")
            col3.metric("RMSE", f"{rmse:.4f}")

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(history_log["loss"],     label="Train Loss")
            ax.plot(history_log["val_loss"], label="Val Loss")
            ax.set_xlabel("Epoch"); ax.set_ylabel("MSE")
            ax.set_title("ANN Training Curve")
            ax.legend()
            st.pyplot(fig)
            plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# TAB 6 — ADVANCED ANALYTICS
# ─────────────────────────────────────────────────────────────────────────────
with tabs[5]:
    st.header("📈 Part 6 — Advanced Analytics")

    # 6.1 High-demand medicines
    st.subheader("🏆 Top 10 High-Demand Medicines")
    high_demand = (
        master.groupby(["medicine_id", "medicine_name"])["quantity"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )
    st.dataframe(high_demand, use_container_width=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.barh(high_demand["medicine_name"], high_demand["quantity"], color="steelblue")
    ax.set_xlabel("Total Units Sold")
    ax.set_title("Top 10 High-Demand Medicines")
    st.pyplot(fig)
    plt.close()

    # 6.2 Inventory shortage
    st.subheader("⚠️ Inventory Shortage Alert")
    med_demand_total = master.groupby("medicine_id")["quantity"].sum().rename("total_sold")
    stocks_cols_needed = [c for c in ["shop_id", "medicine_id", "available_units"]
                          if c in stocks.columns]
    stock_vs_demand = stocks[stocks_cols_needed].merge(
        med_demand_total, on="medicine_id", how="left"
    )
    stock_vs_demand["shortage"] = stock_vs_demand["total_sold"] > stock_vs_demand["available_units"]
    shortages = stock_vs_demand[stock_vs_demand["shortage"]]
    st.metric("Medicines with Potential Shortages", len(shortages))
    if not shortages.empty:
        st.dataframe(shortages.head(20), use_container_width=True)
    else:
        st.success("No shortages detected ✅")

    # 6.3 Customer retention risk
    st.subheader("🚨 Customer Churn Risk (>180 days inactive)")
    latest_date = master["sale_date"].max()
    recency_df = (
        master.groupby("customer_id")["sale_date"]
        .max()
        .reset_index()
        .rename(columns={"sale_date": "last_purchase"})
    )
    recency_df["days_since_purchase"] = (latest_date - recency_df["last_purchase"]).dt.days
    at_risk = recency_df[recency_df["days_since_purchase"] > 180]
    st.metric("At-Risk Customers", len(at_risk))
    if not at_risk.empty:
        st.dataframe(at_risk.sort_values("days_since_purchase", ascending=False).head(20),
                     use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 7 — VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────────────────
with tabs[6]:
    st.header("📉 Part 7 — Visualizations")

    # Monthly revenue trend
    st.subheader("Monthly Revenue Trend")
    monthly_sales = (
        master.groupby(["sale_year", "sale_month"])["total_bill"]
        .sum()
        .reset_index()
        .assign(period=lambda d: pd.to_datetime(
            d["sale_year"].astype(str) + "-" +
            d["sale_month"].astype(str).str.zfill(2)
        ))
        .sort_values("period")
    )
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(monthly_sales["period"], monthly_sales["total_bill"], marker="o", linewidth=1.5)
    ax.set_title("Monthly Revenue Trend")
    ax.set_xlabel("Month"); ax.set_ylabel("Total Revenue (₹)")
    plt.xticks(rotation=45)
    st.pyplot(fig)
    plt.close()

    # Medicine demand heatmap
    st.subheader("Medicine Demand Heatmap (Shop × Category)")
    if isinstance(master["category"], pd.DataFrame):
        master["category"] = master["category"].iloc[:, 0]
    master["category"] = master["category"].astype(str)

    top_cats  = master["category"].value_counts().head(8).index
    top_shops = master["shop_id"].value_counts().head(8).index
    heatmap_data = (
        master[master["category"].isin(top_cats) & master["shop_id"].isin(top_shops)]
        .groupby(["shop_id", "category"])["quantity"]
        .sum()
        .unstack(fill_value=0)
    )
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.heatmap(heatmap_data, annot=True, fmt=".0f", cmap="YlOrRd", ax=ax2)
    ax2.set_title("Medicine Demand Heatmap")
    st.pyplot(fig2)
    plt.close()

    # Category-wise sales
    st.subheader("Sales by Medicine Category")
    cat_sales = master.groupby("category")["total_bill"].sum().sort_values(ascending=False).head(10)
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    cat_sales.plot(kind="bar", ax=ax3, color="teal", edgecolor="black")
    ax3.set_title("Top 10 Categories by Revenue")
    ax3.set_ylabel("Revenue (₹)")
    plt.xticks(rotation=30)
    st.pyplot(fig3)
    plt.close()

st.markdown("---")
st.caption("💊 Pharmacy Chain Analytics System | Built with Streamlit")
