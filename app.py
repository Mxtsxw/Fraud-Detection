import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# --- Configuration ---
st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

# Colors for consistency
C_FRAUD   = "#c0392b"   # muted red — fraud
C_LEGIT   = "#2c5f8a"   # muted blue — legitimate
C_NEUTRAL = "#d0d4e0"   # light gray — background bars
C_AMBER   = "#b7700a"   # amber — warnings
C_GREEN   = "#1e7a4a"   # green — positive signals

PLOTLY_LAYOUT = dict(
    paper_bgcolor="#ffffff",
    plot_bgcolor="#ffffff",
    font=dict(family="Inter", color="#666688", size=11),
    margin=dict(l=10, r=10, t=36, b=10),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
    colorway=[C_LEGIT, C_FRAUD, C_GREEN, C_AMBER, "#7b6ea0"],
)

# Custom CSS
st.markdown("""
<style>
  html, body, [class*="css"] {
      font-family: 'Inter', sans-serif;
      background-color: #f7f8fa;
      color: #1a1a2e;
  }

  [data-testid="metric-container"] {
      background: #ffffff;
      border: 1px solid #e4e6ec;
      border-radius: 10px;
      padding: 16px;
  }
  [data-testid="metric-container"] label {
      color: #888aa0 !important;
      font-size: 0.7rem !important;
      letter-spacing: 0.08em;
      text-transform: uppercase;
  }
  [data-testid="metric-container"] [data-testid="stMetricValue"] {
      font-size: 1.7rem !important;
      font-weight: 700;
      color: #1a1a2e;
  }
  [data-testid="metric-container"] [data-testid="stMetricDelta"] {
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.72rem;
  }

  .section-header {
      font-size: 0.62rem;
      letter-spacing: 0.18em;
      text-transform: uppercase;
      color: #aaaacc;
      border-bottom: 1px solid #e4e6ec;
      padding-bottom: 6px;
      margin: 26px 0 14px;
  }

  .insight-card {
      background: #ffffff;
      border: 1px solid #e4e6ec;
      border-left: 3px solid #c0392b;
      border-radius: 8px;
      padding: 14px 18px;
      margin-bottom: 10px;
      font-size: 0.84rem;
      line-height: 1.65;
      color: #444466;
  }
  .insight-card .tag {
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.63rem;
      background: #fdecea;
      color: #c0392b;
      padding: 2px 8px;
      border-radius: 4px;
      letter-spacing: 0.08em;
      margin-bottom: 6px;
      display: inline-block;
  }
  .insight-card.blue { border-left-color: #2c5f8a; }
  .insight-card.blue .tag { background: #eaf0f7; color: #2c5f8a; }
  .insight-card.amber { border-left-color: #b7700a; }
  .insight-card.amber .tag { background: #fdf3e3; color: #b7700a; }
  .insight-card.green { border-left-color: #1e7a4a; }
  .insight-card.green .tag { background: #eaf5f0; color: #1e7a4a; }

  .dash-title {
      font-size: 1.9rem;
      font-weight: 700;
      letter-spacing: -0.02em;
      color: #1a1a2e;
  }
  .dash-subtitle {
      color: #aaaacc;
      font-size: 0.75rem;
      letter-spacing: 0.12em;
      text-transform: uppercase;
  }
  .status-dot {
      display: inline-block;
      width: 7px;
      height: 7px;
      border-radius: 50%;
      background: #1e7a4a;
      margin-right: 6px;
  }

  #MainMenu, footer { visibility: hidden; }
  header[data-testid="stHeader"] { background: transparent; }
</style>
""", unsafe_allow_html=True)

# --- Data Loading ---
@st.cache_data
def load_data():
    return pd.read_csv("./data/processed/dataset.csv")


df = load_data()

# --- Sidebar Controls ---
st.sidebar.title("Fraud Detection Dashboard")

# Navigation
st.sidebar.subheader("Views")
current_tab = st.sidebar.radio(
    "Select a page:",
    ["Executive Summary", "Fraud Monitoring", "EDA Insights"]
)

st.sidebar.markdown("---")

# Global Filters
st.sidebar.subheader("Filters")

# Transaction Type Filter
available_types = df['type'].unique().tolist()
selected_types = st.sidebar.multiselect(
    "Transaction Type",
    options=available_types,
    default=available_types
)

# Amount Range Filter
min_amt = float(df['amount'].min())
max_amt = float(1_000_000_000)
selected_amount = st.sidebar.slider(
    "Transaction Amount Range",
    min_value=min_amt,
    max_value=max_amt,
    value=(min_amt, max_amt)
)

# --- Data Filtering ---
# Apply filters to create a working dataset for the active tab
filtered_df = df[
    (df['type'].isin(selected_types)) &
    (df['amount'] >= selected_amount[0]) &
    (df['amount'] <= selected_amount[1])
    ]


# --- View Definitions ---

def render_executive_summary(data):
    st.title("Fraud Detection Dashboard")
    st.markdown("Monitoring - Financial Transaction Summary")

    # KPIs
    total_tx = len(data)
    fraud_tx = data['isFraud'].sum() if total_tx > 0 else 0
    fraud_rate = (fraud_tx / total_tx) * 100 if total_tx > 0 else 0
    fraud_amount = data[data['isFraud'] == 1]['amount'].sum() if fraud_tx > 0 else 0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Total Transactions", value=f"{total_tx:,}")
    with col2:
        st.metric(label="Fraudulent Transactions", value=f"{fraud_tx:,}")
    with col3:
        st.metric(label="Fraud Rate", value=f"{fraud_rate:.2f}%")
    with col4:
        st.metric(label="Total Fraud Amount", value=f"${fraud_amount:,.2f}")

    st.markdown("---")

    left_col, center_col, right_col = st.columns([2, 1, 1.5])

    # Bar Chart: Transactions vs Fraud by Type
    with left_col:
        # Prepare data
        type_stats = data.groupby('type').agg(
            Total=('isFraud', 'count'),
            Fraud=('isFraud', 'sum')
        ).reset_index()

        fig_bar = px.bar(
            type_stats,
            x='type',
            y=['Total', 'Fraud'],
            barmode='group',
            # Apply your constants here
            color_discrete_map={'Total': C_LEGIT, 'Fraud': C_FRAUD}
        )

        fig_bar.update_layout(**PLOTLY_LAYOUT, title="Transactions vs Fraud by Type",
                          barmode="overlay", height=280)
        st.plotly_chart(fig_bar, use_container_width=True)

    # 2. Gauge Chart: Fraud Rate
    with center_col:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=fraud_rate,
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, max(1, fraud_rate * 2)]},
                'bar': {'color': "#d62728"},
                "steps": [
                    {"range": [0, 0.1], "color": "#eaf5f0"},
                    {"range": [0.1, 0.5], "color": "#fdf3e3"},
                    {"range": [0.5, 1], "color": "#fdecea"},
                ],
                "threshold": {"value": 0.13, "line": {"color": C_AMBER, "width": 2}}
            },
            title={"text": "Fraud Rate (%)", "font": {"size": 11, "color": "#aaaacc"}},
        ))
        fig_gauge.update_layout(**PLOTLY_LAYOUT, title="Fraud Rate (%)", height=280)
        st.plotly_chart(fig_gauge, use_container_width=True)

    # 3. Table: Last 5 Transactions
    with right_col:
        st.markdown("**Recent Transactions**")
        latest_tx = data.sort_values(by='step', ascending=False).head(5)
        st.table(latest_tx[['type', 'amount']])

    st.markdown("---")

    st.subheader('Temporal Patterns')

    # Logic for time-series analysis
    step_total = data.groupby("step").size()
    step_fraud = data[data["isFraud"] == 1].groupby("step").size().reindex(step_total.index, fill_value=0)
    step_rate = (step_fraud / step_total * 100).fillna(0)
    roll = step_rate.rolling(12).mean()

    fig_time = make_subplots(rows=2, cols=1, shared_xaxes=True,
                             row_heights=[0.4, 0.6], vertical_spacing=0.05)

    # Subplot 1: Fraud Count (Bar)
    fig_time.add_bar(row=1, col=1, x=step_fraud.index, y=step_fraud.values,
                     marker_color=C_FRAUD, name="Fraud count")

    # Subplot 2: Fraud Rate (Line + Markers)
    fig_time.add_scatter(row=2, col=1, x=roll.index, y=roll.values,
                         mode="lines", line=dict(color=C_FRAUD, width=1.5),
                         name="Fraud rate % (12-step avg)")
    fig_time.add_scatter(row=2, col=1, x=step_rate.index, y=step_rate.values,
                         mode="markers", marker=dict(size=2, color=C_FRAUD),
                         name="Fraud rate % (raw)", showlegend=False)

    fig_time.update_layout(**PLOTLY_LAYOUT, title="Fraud Count & Rate over Time (Hourly Steps)",
                           height=350, showlegend=True)

    st.plotly_chart(fig_time, use_container_width=True)

def render_fraud_monitoring(data):
    st.header("Fraud Monitoring")

    fraud_df = df[df["isFraud"] == 1].copy()

    # Top metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Active Fraud Cases", f"{len(fraud_df):,}")
    m2.metric("Avg Fraud Amount", f"${fraud_df['amount'].mean():,.0f}")
    m3.metric("Max Single Fraud", f"${fraud_df['amount'].max():,.0f}")
    m4.metric("Account Drain Rate",
              f"{fraud_df['isOrigDrained'].mean() * 100:.1f}%",
              delta="High risk", delta_color="inverse")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Fraud amount distribution vs legitimate
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=np.log1p(df[df["isFraud"] == 0]["amount"].sample(50_000, random_state=42)),
            name="Legitimate", marker_color=C_LEGIT, opacity=0.6,
            nbinsx=60, histnorm="probability density",
        ))
        fig_dist.add_trace(go.Histogram(
            x=np.log1p(fraud_df["amount"]),
            name="Fraud", marker_color=C_FRAUD, opacity=0.8,
            nbinsx=60, histnorm="probability density",
        ))
        fig_dist.update_layout(**PLOTLY_LAYOUT, title="Amount Distribution: Fraud vs Legitimate (log-scaled)",
                               barmode="overlay", height=300,
                               xaxis_title="log(1+amount)", yaxis_title="Density")
        fig_dist.update_xaxes(gridcolor="#1a1a2e", linecolor="#1a1a2e")
        fig_dist.update_yaxes(gridcolor="#1a1a2e", linecolor="#1a1a2e")
        st.plotly_chart(fig_dist, use_container_width=True)

    with col2:
        fraud_type_counts = fraud_df["type"].value_counts().reset_index()
        fraud_type_counts.columns = ["type", "count"]
        fig_pie = go.Figure(go.Pie(
            labels=fraud_type_counts["type"],
            values=fraud_type_counts["count"],
            hole=0.6,
            marker=dict(colors=[C_FRAUD, "#e07060", "#c87a70"], line=dict(color="#ffffff", width=2)),
        ))
        fig_pie.update_layout(**PLOTLY_LAYOUT, title="Fraud by Transaction Type", height=300,
                              annotations=[dict(text="Fraud<br>Types", x=0.5, y=0.5,
                                                font_size=11, showarrow=False, font_color="#888aa0")])
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("---")

    # Balance error analysis
    st.markdown("<div class='section-header'>Balance Anomaly Detection</div>", unsafe_allow_html=True)
    col3, col4 = st.columns(2)

    with col3:
        sample_f = fraud_df.sample(min(3000, len(fraud_df)), random_state=1)
        sample_n = df[df["isFraud"] == 0].sample(min(3000, len(df[df["isFraud"] == 0])), random_state=1)
        fig_err = go.Figure()
        fig_err.add_trace(go.Scatter(
            x=np.log1p(sample_n["errorBalanceOrig"].abs()),
            y=np.log1p(sample_n["errorBalanceDest"].abs()),
            mode="markers", name="Legitimate",
            marker=dict(color=C_LEGIT, size=3, opacity=0.4),
        ))
        fig_err.add_trace(go.Scatter(
            x=np.log1p(sample_f["errorBalanceOrig"].abs()),
            y=np.log1p(sample_f["errorBalanceDest"].abs()),
            mode="markers", name="Fraud",
            marker=dict(color=C_FRAUD, size=4, opacity=0.7),
        ))
        fig_err.update_layout(**PLOTLY_LAYOUT,
                              title="Balance Error Signature (Origin vs Dest)",
                              xaxis_title="log|errorBalanceOrig|",
                              yaxis_title="log|errorBalanceDest|",
                              height=320)
        fig_err.update_xaxes(gridcolor="#1a1a2e", linecolor="#1a1a2e")
        fig_err.update_yaxes(gridcolor="#1a1a2e", linecolor="#1a1a2e")
        st.plotly_chart(fig_err, use_container_width=True)

    with col4:
        fig_box = go.Figure()
        for label, color, name in [(0, C_LEGIT, "Legitimate"), (1, C_FRAUD, "Fraud")]:
            subset = df[df["isFraud"] == label]["amount"].sample(min(5000, len(df[df["isFraud"] == label])),
                                                                 random_state=2)
            fig_box.add_trace(go.Box(
                y=np.log1p(subset), name=name,
                marker_color=color, line_color=color,
                boxmean=True,
            ))
        fig_box.update_layout(**PLOTLY_LAYOUT, title="Transaction Amount Box Plot (log-scaled)",
                              height=320, yaxis_title="log(1+amount)")
        fig_box.update_xaxes(gridcolor="#1a1a2e", linecolor="#1a1a2e")
        fig_box.update_yaxes(gridcolor="#1a1a2e", linecolor="#1a1a2e")
        st.plotly_chart(fig_box, use_container_width=True)

    # Rule-based detection quality
    st.markdown("<div class='section-header'>Rule-Based Detection Performance</div>", unsafe_allow_html=True)
    tp = df[(df["isFlaggedFraud"] == 1) & (df["isFraud"] == 1)].shape[0]
    fp = df[(df["isFlaggedFraud"] == 1) & (df["isFraud"] == 0)].shape[0]
    fn = df[(df["isFlaggedFraud"] == 0) & (df["isFraud"] == 1)].shape[0]
    tn = df[(df["isFlaggedFraud"] == 0) & (df["isFraud"] == 0)].shape[0]

    prec = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("True Positives", f"{tp:,}")
    c2.metric("False Positives", f"{fp:,}")
    c3.metric("Precision", f"{prec:.1f}%")
    c4.metric("Recall", f"{rec:.2f}%", delta="⚠️ Critical gap", delta_color="inverse")

    fig_conf = go.Figure(go.Heatmap(
        z=[[tn, fp], [fn, tp]],
        x=["Pred: Not Fraud", "Pred: Fraud"],
        y=["Actual: Not Fraud", "Actual: Fraud"],
        text=[[f"{tn:,}", f"{fp:,}"], [f"{fn:,}", f"{tp:,}"]],
        texttemplate="%{text}",
        colorscale=[[0, "#f7f8fa"], [0.5, "#d6e4f0"], [1, C_FRAUD]],
        showscale=False,
    ))
    fig_conf.update_layout(**PLOTLY_LAYOUT, title="Confusion Matrix — Current Rule-Based System",
                           height=280)
    st.plotly_chart(fig_conf, use_container_width=True)


def render_eda_insights(data):
    st.title("EDA Insights")
    st.markdown("Exploratory Data Analysis covering distribution, outliers, and feature relationships.")

    st.markdown("""
        <div class='insight-card'>
          <div class='tag'>CRITICAL · CLASS IMBALANCE</div><br>
          The dataset contains <strong>~0.13% fraud rate</strong> across 6.36M transactions — only 8,213 genuine
          fraud cases. This severe imbalance demands precision/recall/F1 metrics, stratified cross-validation,
          and class-weighted loss functions rather than accuracy-based evaluation.
        </div>
        <div class='insight-card blue'>
          <div class='tag'>STRUCTURAL INSIGHT · TRANSACTION TYPES</div><br>
          Fraud is <strong>exclusively present in TRANSFER and CASH_OUT</strong> transaction types.
          PAYMENT, CASH_IN, and DEBIT have zero fraudulent transactions. This single categorical signal
          is the strongest predictor and suggests type-specific modeling strategies.
        </div>
        <div class='insight-card amber'>
          <div class='tag'>BEHAVIORAL PATTERN · ACCOUNT DRAIN</div><br>
          Approximately <strong>70% of fraud cases completely drain the origin account</strong>
          (<code>newbalanceOrig = 0</code>). The feature <code>isOrigDrained</code> is a high-signal binary
          engineered feature. Many fraudsters transfer the exact account balance — making
          <code>oldbalanceOrg == amount</code> another detectable pattern.
        </div>
        <div class='insight-card green'>
          <div class='tag'>ENGINEERED FEATURES · BALANCE ERRORS</div><br>
          Two balance discrepancy features expose anomalies invisible in raw data:<br>
          • <code>errorBalanceOrig = newbalanceOrig + amount − oldbalanceOrg</code> (should be ≈0)<br>
          • <code>errorBalanceDest = oldbalanceDest + amount − newbalanceDest</code> (should be ≈0)<br>
          Fraud transactions show non-zero error values, especially in destination balance.
        </div>
        <div class='insight-card'>
          <div class='tag'>ACCOUNT TYPES · C vs M</div><br>
          All transaction origins are customer accounts (<code>C</code>). <strong>Fraud destinations
          are predominantly customer-to-customer</strong> transfers — merchant accounts (<code>M</code>)
          have zero balance tracking and are never fraud destinations. <code>destType</code> is a
          strong binary feature for modeling.
        </div>
        <div class='insight-card blue'>
          <div class='tag'>TEMPORAL PATTERN · PERIODICITY</div><br>
          Fraud events show <strong>periodic patterns</strong> across the 743-hour observation window
          (~30 days). Transaction volume decreases in the second half of the period. Engineering
          <code>hour_of_day = step % 24</code> and <code>day = step // 24</code> may capture
          cyclical fraud timing behavior.
        </div>
        <div class='insight-card amber'>
          <div class='tag'>RULE-BASED GAP · isFlaggedFraud</div><br>
          The existing rule-based system (<code>isFlaggedFraud</code>) catches only <strong>0.19% of
          true fraud cases</strong>. This column must be excluded from ML features due to leakage risk
          and its misleading low recall. It demonstrates the clear business need for ML-powered detection.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Feature Engineering Roadmap</div>", unsafe_allow_html=True)

    feature_data = {
        "Feature": ["log_amount", "errorBalanceOrig", "errorBalanceDest", "destType",
                    "type (OHE)", "step / hour_of_day", "isOrigDrained"],
        "Type": ["Numeric", "Numeric", "Numeric", "Binary", "Categorical", "Numeric", "Binary"],
        "Source": ["log1p(amount)", "newbalanceOrig + amount − oldbalanceOrg",
                   "oldbalanceDest + amount − newbalanceDest",
                   "nameDest prefix", "Transaction type", "step as-is or cyclic",
                   "newbalanceOrig == 0"],
        "Signal Strength": ["Medium", "High", "High", "High", "Critical", "Low", "High"],
        "Drop": ["amount (raw)", "—", "—", "nameOrig, nameDest", "—", "—", "—"],
    }
    feat_df = pd.DataFrame(feature_data)
    strength_colors = {"Critical": "🔴", "High": "🟠", "Medium": "🟡", "Low": "⚪"}
    feat_df["Signal Strength"] = feat_df["Signal Strength"].map(lambda x: f"{strength_colors.get(x, '')} {x}")
    st.dataframe(feat_df, use_container_width=True, hide_index=True)

    st.markdown("<div class='section-header'>Recommended Modeling Strategy</div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
            <div style='background:#eaf5f0;border:1px solid #c8e6d8;border-radius:8px;padding:14px'>
              <div style='color:#1e7a4a;font-weight:700;margin-bottom:6px'>✓ Baseline</div>
              <div style='color:#444466;font-size:0.82rem'>Logistic Regression with class weights.
              Establishes precision/recall trade-off baseline. Fast to train and interpretable.</div>
            </div>
            """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
            <div style='background:#eaf0f7;border:1px solid #c8d8ea;border-radius:8px;padding:14px'>
              <div style='color:#2c5f8a;font-weight:700;margin-bottom:6px'>~ Intermediate</div>
              <div style='color:#444466;font-size:0.82rem'>Random Forest Classifier with stratified CV.
              Handles non-linear patterns, provides feature importances for regulatory reporting.</div>
            </div>
            """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
            <div style='background:#fdecea;border:1px solid #ead8d6;border-radius:8px;padding:14px'>
              <div style='color:#c0392b;font-weight:700;margin-bottom:6px'>★ Target</div>
              <div style='color:#444466;font-size:0.82rem'>XGBoost  with AUC-PR optimization.
              Best performance on imbalanced datasets. Can be calibrated probability thresholds.</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Data extracted from the Gini Importance visual
    feat_importance = pd.read_csv("./data/processed/feat_imp_df.csv")
    feat_importance = feat_importance.sort_values("importance", ascending=True)
    fig_imp = go.Figure(go.Bar(
        x=feat_importance["importance"],
        y=feat_importance["feature"],
        orientation="h",
        marker=dict(
            color=feat_importance["importance"],
            colorscale=[[0, "#d6e4f0"], [0.5, "#8ab4d8"], [1, C_FRAUD]],
            showscale=False,
        ),
    ))
    fig_imp.update_layout(**PLOTLY_LAYOUT, title="Feature Importance (from Baseline Random Forest)",
                          height=280, xaxis_title="Relative Importance")
    fig_imp.update_xaxes(gridcolor="#1a1a2e", linecolor="#1a1a2e")
    fig_imp.update_yaxes(gridcolor="#1a1a2e", linecolor="#1a1a2e")
    st.plotly_chart(fig_imp, use_container_width=True)


# --- Routing ---
if current_tab == "Executive Summary":
    render_executive_summary(filtered_df)
elif current_tab == "Fraud Monitoring":
    render_fraud_monitoring(filtered_df)
elif current_tab == "EDA Insights":
    render_eda_insights(filtered_df)