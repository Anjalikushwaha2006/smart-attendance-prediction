import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from datetime import date
from io import BytesIO

st.set_page_config(page_title="Smart Attendance AI", page_icon="🎓", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.3rem; padding-bottom: 2rem;}
.hero {padding: 1.4rem 1.6rem; border-radius: 18px; background: linear-gradient(120deg,#15233f,#315b85); color:white; margin-bottom:1rem;}
.hero h1 {margin:0; font-size:2.1rem;}
.hero p {margin:.4rem 0 0 0; opacity:.9;}
div[data-testid="stMetric"] {background:#f4f7fb; border:1px solid #e2e8f0; padding:14px; border-radius:14px;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero"><h1>🎓 Smart Attendance Prediction System</h1><p>Interactive attendance analytics, next-lecture prediction and early at-risk alerts using Machine Learning.</p></div>', unsafe_allow_html=True)

@st.cache_data
def make_demo_data():
    rng = np.random.default_rng(2026)
    students = [f"STU{i:03d}" for i in range(1, 61)]
    subjects = ["Machine Learning", "Signals & Systems", "Digital Communication", "Python Lab", "Mathematics"]
    slots = [9, 10, 11, 13, 14]
    days = pd.bdate_range("2026-08-03", "2026-10-23")
    rows = []
    student_habit = {s: rng.uniform(-0.17, 0.16) for s in students}
    previous = {s: 1 for s in students}
    history = {s: [] for s in students}
    for d in days:
        for s in students:
            for j, sub in enumerate(subjects):
                slot = slots[j]
                dow = d.dayofweek
                roll = float(np.mean(history[s][-5:])) if history[s] else 0.8
                abs_week = int(sum(1-x for x in history[s][-35:]))
                p = 0.84 + student_habit[s] - (0.055 if slot in (9,14) else 0) - (0.035 if dow == 4 else 0) + 0.12*(roll-0.75)
                p = float(np.clip(p, 0.25, 0.98))
                attended = int(rng.random() < p)
                rows.append({
                    "Student_ID": s, "Subject": sub, "Date": d.date(),
                    "Lecture_Slot": slot, "Prev_Lecture_Status": previous[s],
                    "Rolling_Attendance_5": roll, "Absences_Last_Week": abs_week,
                    "Attended": attended
                })
                previous[s] = attended
                history[s].append(attended)
    df = pd.DataFrame(rows)
    # Demonstrate cleaning in the UI, while keeping the sample data usable.
    return df

def load_data(uploaded):
    if uploaded is None:
        return make_demo_data(), "Built-in demonstration data (synthetic)"
    try:
        df = pd.read_csv(uploaded)
        required = {"Student_ID", "Subject", "Date", "Lecture_Slot", "Attended"}
        missing = required - set(df.columns)
        if missing:
            st.error("Uploaded CSV is missing required columns: " + ", ".join(sorted(missing)))
            st.stop()
        return df, "Uploaded CSV"
    except Exception as e:
        st.error(f"Could not read the CSV: {e}")
        st.stop()

with st.sidebar:
    st.header("⚙️ Controls")
    uploaded = st.file_uploader("Upload attendance.csv (optional)", type=["csv"])
    st.caption("No upload? The dashboard uses clearly labelled synthetic demo data.")
    threshold = st.slider("Attendance shortage threshold (%)", 50, 90, 75, 1)
    alert_cutoff = st.slider("Absence alert probability (%)", 20, 80, 50, 5) / 100
    st.divider()
    st.markdown("**Project:** Smart Attendance Prediction")
    st.markdown("**Models:** Logistic Regression & Random Forest")

raw, data_source = load_data(uploaded)
df = raw.copy()
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Attended"] = pd.to_numeric(df["Attended"], errors="coerce")
df["Lecture_Slot"] = pd.to_numeric(df["Lecture_Slot"], errors="coerce")
if "Prev_Lecture_Status" not in df:
    df["Prev_Lecture_Status"] = df.groupby("Student_ID")["Attended"].shift(1)
if "Rolling_Attendance_5" not in df:
    df["Rolling_Attendance_5"] = df.groupby("Student_ID")["Attended"].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
if "Absences_Last_Week" not in df:
    df["Absences_Last_Week"] = df.groupby("Student_ID")["Attended"].transform(lambda x: (1-x).shift(1).rolling(35, min_periods=1).sum())
df["Day_of_Week"] = df["Date"].dt.dayofweek
before = len(df)
df = df.drop_duplicates().dropna(subset=["Student_ID", "Subject", "Date", "Lecture_Slot", "Attended"])
df = df[df["Attended"].isin([0,1])].copy()
df = df.sort_values(["Date", "Student_ID", "Lecture_Slot"])
df["Attended"] = df["Attended"].astype(int)
df["Prev_Lecture_Status"] = df["Prev_Lecture_Status"].fillna(0)
df["Rolling_Attendance_5"] = df["Rolling_Attendance_5"].fillna(df["Attended"].mean())
df["Absences_Last_Week"] = df["Absences_Last_Week"].fillna(0)
df["Day_of_Week"] = df["Day_of_Week"].fillna(0).astype(int)

if df.empty:
    st.error("No usable rows remain after cleaning. Check your CSV.")
    st.stop()

# Attendance rate by student. The project uses 75% as an assumed threshold; confirm local rules.
student_rates = df.groupby("Student_ID", as_index=False).agg(
    Attendance_Rate=("Attended", "mean"),
    Lectures=("Attended", "size"),
    Absences=("Attended", lambda x: int((1-x).sum()))
)
student_rates["Attendance_%"] = 100 * student_rates["Attendance_Rate"]
student_rates["Status"] = np.where(student_rates["Attendance_%"] < threshold, "At Risk", "On Track")

# Time-based holdout: latest 20% of dates for testing, to avoid random temporal leakage.
unique_dates = sorted(df["Date"].dropna().unique())
if len(unique_dates) >= 5:
    cutoff_idx = max(1, int(len(unique_dates)*0.8))
    cutoff_date = pd.Timestamp(unique_dates[cutoff_idx])
    train_df = df[df["Date"] < cutoff_date].copy()
    test_df = df[df["Date"] >= cutoff_date].copy()
else:
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["Attended"] if df["Attended"].nunique()>1 else None)

num_features = ["Lecture_Slot", "Day_of_Week", "Prev_Lecture_Status", "Rolling_Attendance_5", "Absences_Last_Week"]
cat_features = ["Subject"]
feature_cols = num_features + cat_features
if train_df["Attended"].nunique() < 2:
    st.error("Training data has only one attendance class. Upload a larger dataset with both present (1) and absent (0) examples.")
    st.stop()

def build_model(model_kind):
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    pre = ColumnTransformer([("num", numeric_pipe, num_features), ("cat", categorical_pipe, cat_features)])
    clf = LogisticRegression(max_iter=1000, class_weight="balanced") if model_kind == "Logistic Regression" else RandomForestClassifier(n_estimators=180, max_depth=12, min_samples_leaf=3, class_weight="balanced", random_state=42, n_jobs=-1)
    return Pipeline([("preprocess", pre), ("model", clf)])

@st.cache_resource(show_spinner="Training attendance models...")
def train_models(train_frame, model_features):
    fitted = {}
    for kind in ["Logistic Regression", "Random Forest"]:
        m = build_model(kind)
        m.fit(train_frame[model_features], train_frame["Attended"])
        fitted[kind] = m
    return fitted

# cache key-safe training input is prepared as a dataframe copy
try:
    models = train_models(train_df[feature_cols + ["Attended"]].copy(), feature_cols)
except Exception as e:
    st.error(f"Model training failed: {e}")
    st.stop()

predictions = {}
metrics = []
for name, model in models.items():
    pred = model.predict(test_df[feature_cols])
    predictions[name] = pred
    metrics.append({
        "Model": name,
        "Accuracy": accuracy_score(test_df["Attended"], pred),
        "Precision": precision_score(test_df["Attended"], pred, zero_division=0),
        "Recall": recall_score(test_df["Attended"], pred, zero_division=0),
        "F1-score": f1_score(test_df["Attended"], pred, zero_division=0)
    })
metric_df = pd.DataFrame(metrics)

pages = ["Overview", "Attendance Explorer", "At-Risk Alerts", "Predict Next Lecture", "Model Performance", "Project Guide"]
page = st.sidebar.radio("Navigate", pages)

if page == "Overview":
    st.subheader("📊 Dashboard Overview")
    st.caption(f"Data source: {data_source} · Cleaned rows: {len(df):,} · Removed duplicates/invalid rows: {before-len(df):,}")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Students", f"{df['Student_ID'].nunique():,}")
    c2.metric("Attendance records", f"{len(df):,}")
    c3.metric("Overall attendance", f"{df['Attended'].mean()*100:.1f}%")
    c4.metric("Students at risk", f"{(student_rates['Attendance_%'] < threshold).sum():,}")
    left,right = st.columns(2)
    with left:
        by_subject = df.groupby("Subject", as_index=False)["Attended"].mean()
        by_subject["Attendance (%)"] = by_subject["Attended"]*100
        fig = px.bar(by_subject, x="Subject", y="Attendance (%)", color="Attendance (%)", title="Attendance by Subject", text_auto=".1f")
        fig.update_layout(template="plotly_white", yaxis_range=[0,100], xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        daily = df.groupby("Date", as_index=False)["Attended"].mean()
        daily["Attendance (%)"] = daily["Attended"]*100
        fig = px.line(daily, x="Date", y="Attendance (%)", title="Attendance Trend Over Time", markers=True)
        fig.update_layout(template="plotly_white", yaxis_range=[0,100])
        st.plotly_chart(fig, use_container_width=True)
    st.subheader("Students needing attention")
    st.dataframe(student_rates.sort_values("Attendance_%").head(10).style.format({"Attendance_Rate":"{:.1%}","Attendance_%":"{:.1f}%"}), use_container_width=True, hide_index=True)

elif page == "Attendance Explorer":
    st.subheader("🔎 Attendance Explorer")
    a,b,c = st.columns(3)
    with a:
        selected_students = st.multiselect("Students", sorted(df["Student_ID"].astype(str).unique()), default=[])
    with b:
        selected_subjects = st.multiselect("Subjects", sorted(df["Subject"].astype(str).unique()), default=[])
    with c:
        min_date, max_date = df["Date"].min().date(), df["Date"].max().date()
        date_range = st.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    view = df.copy()
    if selected_students: view = view[view["Student_ID"].astype(str).isin(selected_students)]
    if selected_subjects: view = view[view["Subject"].astype(str).isin(selected_subjects)]
    if isinstance(date_range, tuple) and len(date_range)==2:
        view = view[view["Date"].dt.date.between(date_range[0], date_range[1])]
    st.write(f"Matching records: **{len(view):,}**")
    x,y = st.columns(2)
    with x:
        subject_plot = view.groupby("Subject", as_index=False)["Attended"].mean()
        subject_plot["Attendance (%)"] = subject_plot["Attended"]*100
        st.plotly_chart(px.bar(subject_plot, x="Subject", y="Attendance (%)", title="Filtered attendance by subject", range_y=[0,100]), use_container_width=True)
    with y:
        slot_plot = view.groupby("Lecture_Slot", as_index=False)["Attended"].mean()
        slot_plot["Attendance (%)"] = slot_plot["Attended"]*100
        st.plotly_chart(px.bar(slot_plot, x="Lecture_Slot", y="Attendance (%)", title="Attendance by lecture slot", range_y=[0,100]), use_container_width=True)
    st.dataframe(view.drop(columns=["Day_of_Week"], errors="ignore"), use_container_width=True, hide_index=True)
    st.download_button("⬇️ Download filtered CSV", view.to_csv(index=False).encode("utf-8"), "filtered_attendance.csv", "text/csv")

elif page == "At-Risk Alerts":
    st.subheader("🚨 Early At-Risk Alerts")
    st.write(f"Students below the selected attendance threshold of **{threshold}%** are flagged for follow-up.")
    risk = student_rates[student_rates["Attendance_%"] < threshold].sort_values("Attendance_%")
    if risk.empty:
        st.success("No students are currently below the selected threshold.")
    else:
        st.warning(f"{len(risk)} student(s) need attention.")
        st.dataframe(risk.style.format({"Attendance_Rate":"{:.1%}","Attendance_%":"{:.1f}%"}), use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download at-risk list", risk.to_csv(index=False).encode("utf-8"), "at_risk_students.csv", "text/csv")
    st.caption("This is a support alert based on observed attendance rate, not a disciplinary decision.")

elif page == "Predict Next Lecture":
    st.subheader("🔮 Predict Next-Lecture Attendance")
    st.info("Choose a student and lecture details. The selected model estimates Present/Absent using learned patterns.")
    c1,c2,c3 = st.columns(3)
    with c1:
        student = st.selectbox("Student", sorted(df["Student_ID"].astype(str).unique()))
        hist = df[df["Student_ID"].astype(str)==student].sort_values("Date")
        rate = float(hist["Attended"].mean()) if len(hist) else float(df["Attended"].mean())
        st.metric("Student's observed attendance", f"{rate*100:.1f}%")
    with c2:
        subject = st.selectbox("Next subject", sorted(df["Subject"].astype(str).unique()))
        slot = st.selectbox("Lecture slot", sorted(df["Lecture_Slot"].dropna().unique().tolist()))
    with c3:
        day = st.selectbox("Day of week", ["Monday","Tuesday","Wednesday","Thursday","Friday"], index=0)
        model_name = st.selectbox("Prediction model", ["Random Forest", "Logistic Regression"])
    day_number = ["Monday","Tuesday","Wednesday","Thursday","Friday"].index(day)
    last = hist.iloc[-1] if len(hist) else df.iloc[-1]
    rolling = float(hist["Attended"].tail(5).mean()) if len(hist) else float(df["Attended"].mean())
    absences = int((1-hist["Attended"].tail(35)).sum()) if len(hist) else 0
    row = pd.DataFrame([{
        "Lecture_Slot": float(slot), "Day_of_Week": day_number,
        "Prev_Lecture_Status": int(last["Attended"]),
        "Rolling_Attendance_5": rolling, "Absences_Last_Week": absences,
        "Subject": subject
    }])
    model = models[model_name]
    p_attend = float(model.predict_proba(row[feature_cols])[0][list(model.named_steps["model"].classes_).index(1)])
    p_absent = 1-p_attend
    pred = "Present" if p_attend >= 0.5 else "Absent"
    m1,m2,m3 = st.columns(3)
    m1.metric("Prediction", pred)
    m2.metric("Probability of attendance", f"{p_attend*100:.1f}%")
    m3.metric("Probability of absence", f"{p_absent*100:.1f}%")
    fig = go.Figure(go.Bar(x=["Present","Absent"], y=[p_attend*100,p_absent*100], text=[f"{p_attend*100:.1f}%",f"{p_absent*100:.1f}%"], textposition="auto"))
    fig.update_layout(title="Model confidence (estimated probabilities)", yaxis_title="Probability (%)", yaxis_range=[0,100], template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)
    if p_absent >= alert_cutoff:
        st.warning("Early alert: elevated predicted absence probability. Consider a supportive check-in.")
    else:
        st.success("No alert at the selected probability threshold.")
    st.caption("Predictions are educational estimates from historical/synthetic data, not guarantees.")

elif page == "Model Performance":
    st.subheader("🧪 Model Performance (time-based test set)")
    st.caption(f"Training records: {len(train_df):,} · Test records: {len(test_df):,} · Test period begins {test_df['Date'].min().date() if len(test_df) else 'N/A'}")
    st.dataframe(metric_df.style.format({c:"{:.3f}" for c in ["Accuracy","Precision","Recall","F1-score"]}), use_container_width=True, hide_index=True)
    plot_df = metric_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
    st.plotly_chart(px.bar(plot_df, x="Metric", y="Score", color="Model", barmode="group", range_y=[0,1], title="Model comparison"), use_container_width=True)
    for model_name in ["Logistic Regression", "Random Forest"]:
        with st.expander(f"Confusion matrix — {model_name}"):
            cm = confusion_matrix(test_df["Attended"], predictions[model_name], labels=[0,1])
            fig = px.imshow(cm, text_auto=True, x=["Predicted Absent","Predicted Present"], y=["Actual Absent","Actual Present"], color_continuous_scale="Blues", title=f"{model_name}: actual vs predicted")
            st.plotly_chart(fig, use_container_width=True)
    rf = models["Random Forest"].named_steps["model"]
    pre = models["Random Forest"].named_steps["preprocess"]
    try:
        names = pre.get_feature_names_out()
        imp = pd.DataFrame({"Feature": names, "Importance": rf.feature_importances_}).sort_values("Importance", ascending=False).head(12)
        st.plotly_chart(px.bar(imp.sort_values("Importance"), x="Importance", y="Feature", orientation="h", title="Random Forest feature importance"), use_container_width=True)
    except Exception:
        st.caption("Feature importance is unavailable for this model configuration.")

else:
    st.subheader("📘 Project Guide")
    st.markdown("""
**Problem statement:** Manual attendance monitoring is time-consuming and attendance shortages may be detected late.

**Objectives**
- Explore attendance by student, subject, date and lecture slot.
- Predict the next lecture's Present/Absent status.
- Compare Logistic Regression with Random Forest.
- Raise early support alerts for students below a configurable attendance threshold.

**Workflow:** CSV → data cleaning → past-history features → chronological train/test split → model training → evaluation → prediction and alert.

**Input columns:** `Student_ID`, `Subject`, `Date`, `Lecture_Slot`, `Attended`. Optional precomputed columns: `Prev_Lecture_Status`, `Rolling_Attendance_5`, `Absences_Last_Week`.

**Important:** If you use the built-in dataset, it is synthetic demonstration data, not real student records. Do not present demo results as verified real-world performance. Confirm the college's official attendance threshold before using alerts.
""")
    st.download_button("⬇️ Download demo dataset", df.to_csv(index=False).encode("utf-8"), "attendance_demo.csv", "text/csv")

st.divider()
st.caption("Smart Attendance Prediction System • Streamlit + Pandas + Plotly + Scikit-learn")
