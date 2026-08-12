# ==============================================================================
# SECTION 1: IMPORTS
# ==============================================================================
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# NOTE: best_model.pkl now contains a trained Keras/TensorFlow model object
# (the notebook's Deep Learning model, dnn_model), not a scikit-learn model.
# TensorFlow must be importable in this environment so that joblib/pickle can
# correctly reconstruct the Keras object when best_model.pkl is loaded below.
try:
    import tensorflow as tf  # noqa: F401
except ImportError:
    st.error(
        "❌ TensorFlow is not installed, but best_model.pkl contains a "
        "trained Deep Learning (Keras) model and requires it. "
        "Please add `tensorflow` to requirements.txt."
    )
    st.stop()
 
# ==============================================================================
# SECTION 2: PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="Used Car Price Predictor | AI Valuation Engine",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)
 
# ==============================================================================
# SECTION 3: FILE PATHS (all files must live in the same project directory)
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
ENCODERS_PATH = os.path.join(BASE_DIR, "encoders.pkl")
DATA_PATH = os.path.join(BASE_DIR, "used_cars.csv")
 
# Columns as defined by the dataset / training pipeline
TARGET_COL = "price"
CATEGORICAL_COLS = [
    "brand", "model", "fuel_type", "engine",
    "transmission", "ext_col", "int_col", "accident", "clean_title",
]
NUMERIC_COLS = ["model_year", "milage"]
ALL_FEATURE_COLS = [
    "brand", "model", "model_year", "milage", "fuel_type",
    "engine", "transmission", "ext_col", "int_col", "accident", "clean_title",
]
 
# ==============================================================================
# SECTION 4: CUSTOM CSS - MODERN / DARK-MODE-FRIENDLY SAAS STYLING
# ==============================================================================
CUSTOM_CSS = """
<style>
    /* ---------- Global font & background ---------- */
    html, body, [class*="css"] {
        font-family: 'Segoe UI', 'Inter', sans-serif;
    }
 
    /* ---------- Hero section ---------- */
    .hero-container {
        padding: 2.2rem 2rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
        color: #ffffff;
        margin-bottom: 1.8rem;
        box-shadow: 0 8px 24px rgba(37, 117, 252, 0.25);
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
    }
    .hero-subtitle {
        font-size: 1.02rem;
        opacity: 0.92;
    }
 
    /* ---------- Section headers ---------- */
    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        margin-top: 1.2rem;
        margin-bottom: 0.6rem;
        border-left: 5px solid #2575fc;
        padding-left: 0.6rem;
    }
 
    /* ---------- Prediction Result Card ---------- */
    .prediction-card {
        padding: 2rem;
        border-radius: 20px;
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        text-align: center;
        color: #ffffff;
        box-shadow: 0 10px 30px rgba(17, 153, 142, 0.35);
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .prediction-label {
        font-size: 1.1rem;
        font-weight: 500;
        opacity: 0.9;
    }
    .prediction-value {
        font-size: 3rem;
        font-weight: 900;
        letter-spacing: 1px;
    }
    .confidence-badge {
        display: inline-block;
        margin-top: 0.6rem;
        padding: 0.35rem 0.9rem;
        border-radius: 30px;
        background: rgba(255,255,255,0.2);
        font-size: 0.9rem;
    }
 
    /* ---------- Generic info card ---------- */
    .info-card {
        padding: 1.2rem;
        border-radius: 14px;
        background: rgba(120, 120, 120, 0.08);
        border: 1px solid rgba(120, 120, 120, 0.18);
        margin-bottom: 0.8rem;
    }
 
    /* ---------- Buttons ---------- */
    div.stButton > button {
        border-radius: 10px;
        font-weight: 600;
        padding: 0.55rem 1.2rem;
        transition: all 0.2s ease-in-out;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 14px rgba(0,0,0,0.15);
    }
 
    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        border-right: 1px solid rgba(120,120,120,0.15);
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
 
# ==============================================================================
# SECTION 5: CACHED LOADERS FOR MODEL, SCALER, ENCODERS, AND DATASET
# ==============================================================================
@st.cache_resource(show_spinner=False)
def load_artifact(path: str):
    """
    Generic loader used for the model, scaler, and encoders.
    NOTE: These artifacts are saved with joblib.dump() in the training notebook,
    so they must be loaded with joblib.load() (not pickle.load()) to avoid the
    'STACK_GLOBAL requires str' deployment error.
    """
    return joblib.load(path)
 
 
@st.cache_data(show_spinner=False)
def load_dataset(path: str) -> pd.DataFrame:
    """
    Loads the reference dataset used to power dependent dropdowns, the
    Market Insights charts, and the min/max/median values behind the
    Mileage input.
 
    NOTE (Bug fix): The raw CSV stores 'price' as text (e.g. "$10,300") and
    'milage' as text (e.g. "99,999 mi."). Left uncleaned, this causes
    `df.groupby("brand")["price"].mean()` to raise
    `TypeError: dtype 'str' does not support operation 'mean'` in the Market
    Insights section. Both columns are cleaned to numeric right here,
    immediately after loading, so every downstream operation (charts,
    aggregates, number_input min/max/value) always receives clean numeric
    data.
    """
    data = pd.read_csv(path)
 
    # ---- Clean the 'price' column: strip '$' and ',' then convert to numeric ----
    if TARGET_COL in data.columns:
        data[TARGET_COL] = (
            data[TARGET_COL]
            .astype(str)
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
        )
        data[TARGET_COL] = pd.to_numeric(data[TARGET_COL], errors="coerce")
 
    # ---- Clean the 'milage' column: strip everything except digits and '.' ----
    if "milage" in data.columns:
        data["milage"] = (
            data["milage"]
            .astype(str)
            .str.replace(r"[^\d.]", "", regex=True)
        )
        data["milage"] = pd.to_numeric(data["milage"], errors="coerce")
 
        # ---- Handle invalid/NaN mileage safely without altering feature structure ----
        if data["milage"].isna().any():
            median_milage = data["milage"].median()
            data["milage"] = data["milage"].fillna(median_milage)
 
    return data
 
 
def load_all_assets():
    """
    Attempts to load every required artifact (model, scaler, encoders, dataset).
    Any missing/corrupted file is caught and surfaced as a clear Streamlit error,
    then the app execution is stopped gracefully.
    """
    assets = {}
    try:
        assets["model"] = load_artifact(MODEL_PATH)
    except FileNotFoundError:
        st.error(f"❌ Model file not found at: `{MODEL_PATH}`. Please add **best_model.pkl**.")
        st.stop()
    except Exception as e:
        st.error(
            "❌ Failed to load the model file. "
            "Please make sure best_model.pkl was created using joblib.dump(). "
            f"Details: {e}"
        )
        st.stop()
 
    try:
        assets["scaler"] = load_artifact(SCALER_PATH)
    except FileNotFoundError:
        st.error(f"❌ Scaler file not found at: `{SCALER_PATH}`. Please add **scaler.pkl**.")
        st.stop()
    except Exception as e:
        st.error(
            "❌ Failed to load the scaler file. "
            "Please make sure scaler.pkl was created using joblib.dump(). "
            f"Details: {e}"
        )
        st.stop()
 
    try:
        assets["encoders"] = load_artifact(ENCODERS_PATH)
    except FileNotFoundError:
        st.error(f"❌ Encoders file not found at: `{ENCODERS_PATH}`. Please add **encoders.pkl**.")
        st.stop()
    except Exception as e:
        st.error(
            "❌ Failed to load the encoders file. "
            "Please make sure encoders.pkl was created using joblib.dump(). "
            f"Details: {e}"
        )
        st.stop()
 
    try:
        assets["data"] = load_dataset(DATA_PATH)
    except FileNotFoundError:
        st.error(f"❌ Dataset not found at: `{DATA_PATH}`. Please add **used_cars.csv**.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Failed to load the dataset. Details: {e}")
        st.stop()
 
    return assets
 
 
assets = load_all_assets()
model = assets["model"]
scaler = assets["scaler"]
encoders = assets["encoders"]
df = assets["data"]
 
# ==============================================================================
# SECTION 6: HELPER FUNCTIONS
# ==============================================================================
def get_expected_feature_order():
    """
    Determines the exact feature name order the model/scaler were trained with,
    preventing the classic error:
    'ValueError: Feature names should match those passed during fit.'
 
    Priority:
        1. scaler.feature_names_in_  (most reliable, since scaler runs last before predict)
        2. model.feature_names_in_
        3. Fallback to the dataset's own column order (minus the target column)
    """
    if hasattr(scaler, "feature_names_in_"):
        return list(scaler.feature_names_in_)
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    return [c for c in df.columns if c != TARGET_COL]
 
 
def get_encoder_classes(col_name: str):
    """Safely retrieves the original text classes known by a column's LabelEncoder."""
    if col_name in encoders:
        return list(encoders[col_name].classes_)
    # Fallback to unique dataset values if an encoder is missing for a column
    return sorted(df[col_name].dropna().unique().tolist())
 
 
def encode_value(col_name: str, raw_value):
    """
    Transforms a raw text value into its encoded numeric form using the saved
    LabelEncoder. Never fits a new encoder here — only the existing, trained
    encoder classes are used. If the value is not among the classes the
    encoder was trained on, a clear ValueError is raised (caught by the
    prediction try/except) instead of crashing the app.
    """
    if col_name in encoders:
        encoder = encoders[col_name]
        if raw_value not in encoder.classes_:
            raise ValueError(
                f"'{raw_value}' is not a recognized value for '{col_name}'. "
                f"Please choose one of the available dropdown options."
            )
        return int(encoder.transform([raw_value])[0])
    return raw_value
 
 
def build_feature_dataframe(user_inputs: dict) -> pd.DataFrame:
    """
    Builds a single-row DataFrame from user selections, encodes categorical
    columns, and reorders columns to exactly match the training feature order.
    """
    encoded_row = {}
    for col in ALL_FEATURE_COLS:
        value = user_inputs[col]
        if col in CATEGORICAL_COLS:
            encoded_row[col] = encode_value(col, value)
        else:
            encoded_row[col] = value
 
    row_df = pd.DataFrame([encoded_row])
 
    # Reorder / align columns exactly as expected by the scaler & model
    expected_order = get_expected_feature_order()
    row_df = row_df.reindex(columns=expected_order)
 
    return row_df
 
 
def format_price(value: float) -> str:
    """Formats a numeric price as a currency string, e.g. $24,532.18"""
    return f"${value:,.2f}"


def predict_price(scaled_features) -> float:
    """
    Runs a prediction using the loaded model and safely converts the result
    to a plain Python float.

    The best model produced by the training notebook is a trained Keras/
    TensorFlow Deep Learning model (dnn_model), which:
      - accepts an optional `verbose` kwarg (scikit-learn estimators do not,
        so that case is caught and retried without it), and
      - returns a 2D numpy array (e.g. shape (1, 1)) rather than a plain
        Python float.
    This helper never assumes model.predict() returns a scalar.
    """
    try:
        raw_prediction = model.predict(scaled_features, verbose=0)
    except TypeError:
        # scikit-learn estimators don't accept a 'verbose' kwarg
        raw_prediction = model.predict(scaled_features)
    return float(np.asarray(raw_prediction).reshape(-1)[0])
 
 
def estimate_confidence(feature_row: pd.DataFrame):
    """
    Estimates a prediction confidence range when the underlying model supports it
    (e.g., ensembles with individual estimators such as RandomForest/GradientBoosting).
    Returns (lower_bound, upper_bound, std_dev) or None if unsupported.
    """
    try:
        if hasattr(model, "estimators_"):
            scaled_row = scaler.transform(feature_row)
            preds = np.array([est.predict(scaled_row)[0] for est in model.estimators_])
            mean_pred = preds.mean()
            std_pred = preds.std()
            return mean_pred - std_pred, mean_pred + std_pred, std_pred
    except Exception:
        return None
    return None
 
 
# ==============================================================================
# SECTION 7: SESSION STATE INITIALIZATION (for the Reset button)
# ==============================================================================
if "reset_trigger" not in st.session_state:
    st.session_state.reset_trigger = 0
 
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
 
 
def reset_form():
    """Clears the prediction result and forces widgets to reset to defaults."""
    st.session_state.prediction_result = None
    st.session_state.reset_trigger += 1
 
 
# ==============================================================================
# SECTION 8: SIDEBAR - PROJECT INFORMATION
# ==============================================================================
with st.sidebar:
    # ---- Logo placeholder ----
    st.markdown(
        """
        <div style="text-align:center; padding: 0.5rem 0 1rem 0;">
            <div style="
                width: 90px; height: 90px; margin: 0 auto;
                border-radius: 50%;
                background: linear-gradient(135deg, #6a11cb, #2575fc);
                display: flex; align-items: center; justify-content: center;
                font-size: 2.2rem;">
                🚘
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
 
    st.markdown("## 📌 Project Information")
    st.markdown(
        """
        **Project Name:** Used Car Price Predictor
        **Model:** Best Trained Regression Model (Scikit-Learn)
        **Developer:** Your Name
        **Purpose:** Graduation / ML Course Project
        """
    )
 
    st.markdown("---")
    st.markdown("## 📊 Dataset Information")
    st.markdown(f"- **Rows:** {df.shape[0]:,}")
    st.markdown(f"- **Columns:** {df.shape[1]}")
    st.markdown(f"- **Unique Brands:** {df['brand'].nunique()}")
    st.markdown(f"- **Target Variable:** `price`")
 
    st.markdown("---")
    st.markdown("## ⚙️ Tech Stack")
    st.markdown("`Python` `Scikit-Learn` `Streamlit` `Plotly` `Pandas`")
 
    st.markdown("---")
    st.caption("© 2026 Used Car AI Valuation Engine. All rights reserved.")
 
# ==============================================================================
# SECTION 9: HERO SECTION
# ==============================================================================
st.markdown(
    """
    <div class="hero-container">
        <div class="hero-title">🚗 AI-Powered Used Car Price Predictor</div>
        <div class="hero-subtitle">
            Get an instant, data-driven valuation for any used car using a
            professionally trained Machine Learning model.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
 
# ==============================================================================
# SECTION 10: INPUT FORM - VEHICLE DETAILS (TWO-COLUMN LAYOUT)
# ==============================================================================
st.markdown('<div class="section-header">🧾 Vehicle Details</div>', unsafe_allow_html=True)
 
col1, col2 = st.columns(2)
 
# ---- COLUMN 1 ----
with col1:
    st.markdown("#### 🏷️ Brand & Model")
 
    brand_options = sorted(df["brand"].dropna().unique().tolist())
    brand = st.selectbox(
        "Brand",
        options=["-- Select Brand --"] + brand_options,
        key=f"brand_{st.session_state.reset_trigger}",
    )
 
    # ---- Dependent Model dropdown: filtered by the selected brand ----
    if brand and brand != "-- Select Brand --":
        model_options = sorted(
            df.loc[df["brand"] == brand, "model"].dropna().unique().tolist()
        )
    else:
        model_options = []
 
    car_model = st.selectbox(
        "Model",
        options=["-- Select Model --"] + model_options,
        key=f"model_{st.session_state.reset_trigger}",
        disabled=(brand == "-- Select Brand --" or not brand),
    )
 
    st.markdown("#### 📅 Year & Mileage")
    model_year = st.number_input(
        "Model Year",
        min_value=int(df["model_year"].min()),
        max_value=int(df["model_year"].max()) + 1,
        value=int(df["model_year"].median()),
        step=1,
        key=f"year_{st.session_state.reset_trigger}",
    )
    milage = st.number_input(
        "Mileage (miles)",
        min_value=0,
        max_value=int(df["milage"].max()) * 2,
        value=int(df["milage"].median()),
        step=500,
        key=f"milage_{st.session_state.reset_trigger}",
    )
 
    st.markdown("#### ⛽ Fuel & Engine")
    fuel_type = st.selectbox(
        "Fuel Type",
        options=["-- Select Fuel Type --"] + get_encoder_classes("fuel_type"),
        key=f"fuel_{st.session_state.reset_trigger}",
    )
    engine = st.selectbox(
        "Engine",
        options=["-- Select Engine --"] + get_encoder_classes("engine"),
        key=f"engine_{st.session_state.reset_trigger}",
    )
 
# ---- COLUMN 2 ----
with col2:
    st.markdown("#### ⚙️ Transmission & Colors")
    transmission = st.selectbox(
        "Transmission",
        options=["-- Select Transmission --"] + get_encoder_classes("transmission"),
        key=f"transmission_{st.session_state.reset_trigger}",
    )
    ext_col = st.selectbox(
        "Exterior Color",
        options=["-- Select Exterior Color --"] + get_encoder_classes("ext_col"),
        key=f"ext_col_{st.session_state.reset_trigger}",
    )
    int_col = st.selectbox(
        "Interior Color",
        options=["-- Select Interior Color --"] + get_encoder_classes("int_col"),
        key=f"int_col_{st.session_state.reset_trigger}",
    )
 
    st.markdown("#### 🛡️ History & Title")
    accident = st.selectbox(
        "Accident History",
        options=["-- Select Accident Status --"] + get_encoder_classes("accident"),
        key=f"accident_{st.session_state.reset_trigger}",
    )
    clean_title = st.selectbox(
        "Clean Title",
        options=["-- Select Clean Title Status --"] + get_encoder_classes("clean_title"),
        key=f"clean_title_{st.session_state.reset_trigger}",
    )
 
    # ---- Vehicle summary preview card ----
    with st.expander("🔍 Preview Vehicle Summary", expanded=False):
        st.write(
            {
                "Brand": brand,
                "Model": car_model,
                "Year": model_year,
                "Mileage": f"{milage:,} mi",
                "Fuel Type": fuel_type,
                "Engine": engine,
                "Transmission": transmission,
                "Exterior Color": ext_col,
                "Interior Color": int_col,
                "Accident": accident,
                "Clean Title": clean_title,
            }
        )
 
# ==============================================================================
# SECTION 11: ACTION BUTTONS (PREDICT / RESET)
# ==============================================================================
st.markdown("---")
btn_col1, btn_col2 = st.columns([3, 1])
with btn_col1:
    predict_clicked = st.button("🔮 Predict Price", use_container_width=True, type="primary")
with btn_col2:
    st.button("♻️ Reset", use_container_width=True, on_click=reset_form)
 
# ==============================================================================
# SECTION 12: INPUT VALIDATION + PREDICTION LOGIC
# ==============================================================================
if predict_clicked:
    # ---- Validate that every required field has been selected ----
    required_fields = {
        "Brand": brand,
        "Model": car_model,
        "Fuel Type": fuel_type,
        "Engine": engine,
        "Transmission": transmission,
        "Exterior Color": ext_col,
        "Interior Color": int_col,
        "Accident History": accident,
        "Clean Title": clean_title,
    }
    missing_fields = [
        name for name, val in required_fields.items()
        if not val or str(val).startswith("-- Select")
    ]
 
    if missing_fields:
        st.warning(
            "⚠️ Please complete the following field(s) before predicting: "
            + ", ".join(missing_fields)
        )
    else:
        with st.spinner("🤖 Analyzing vehicle data and predicting price..."):
            try:
                # ---- Assemble raw user inputs ----
                user_inputs = {
                    "brand": brand,
                    "model": car_model,
                    "model_year": model_year,
                    "milage": milage,
                    "fuel_type": fuel_type,
                    "engine": engine,
                    "transmission": transmission,
                    "ext_col": ext_col,
                    "int_col": int_col,
                    "accident": accident,
                    "clean_title": clean_title,
                }
 
                # ---- Build correctly-ordered, encoded feature DataFrame ----
                feature_row = build_feature_dataframe(user_inputs)
 
                # ---- Scale features using the saved scaler ----
                scaled_features = scaler.transform(feature_row)
 
                # ---- Predict using the trained model ----
                predicted_price = predict_price(scaled_features)
 
                # ---- Estimate confidence range, if supported by the model ----
                confidence = estimate_confidence(feature_row)
 
                st.session_state.prediction_result = {
                    "price": predicted_price,
                    "confidence": confidence,
                    "inputs": user_inputs,
                }
 
                st.success("✅ Prediction completed successfully!")
                st.balloons()
 
            except ValueError as ve:
                st.error(
                    "❌ A data formatting issue occurred while predicting. "
                    f"Details: {ve}"
                )
            except Exception as e:
                st.error(f"❌ An unexpected error occurred during prediction: {e}")
 
# ==============================================================================
# SECTION 13: DISPLAY PREDICTION RESULT CARD
# ==============================================================================
if st.session_state.prediction_result:
    result = st.session_state.prediction_result
    price_value = result["price"]
    confidence = result["confidence"]
 
    st.markdown('<div class="section-header">💰 Predicted Price</div>', unsafe_allow_html=True)
 
    confidence_html = ""
    if confidence:
        low, high, std = confidence
        confidence_html = (
            f'<div class="confidence-badge">'
            f"Estimated range: {format_price(max(low,0))} – {format_price(high)}"
            f"</div>"
        )
 
    st.markdown(
        f"""
        <div class="prediction-card">
            <div class="prediction-label">Estimated Market Value</div>
            <div class="prediction-value">{format_price(price_value)}</div>
            {confidence_html}
        </div>
        """,
        unsafe_allow_html=True,
    )
 
    # ---- Vehicle info recap in expandable card ----
    with st.expander("📋 View Full Vehicle Report", expanded=False):
        report_cols = st.columns(2)
        items = list(result["inputs"].items())
        half = len(items) // 2
        for c, chunk in zip(report_cols, [items[:half], items[half:]]):
            with c:
                for k, v in chunk:
                    st.markdown(f"**{k.replace('_', ' ').title()}:** {v}")
 
# ==============================================================================
# SECTION 14: MARKET INSIGHTS - PLOTLY CHARTS
# ==============================================================================
st.markdown("---")
st.markdown('<div class="section-header">📈 Market Insights</div>', unsafe_allow_html=True)
 
chart_col1, chart_col2 = st.columns(2)
 
with chart_col1:
    st.markdown("##### Average Price by Brand")
    brand_avg = (
        df.groupby("brand")[TARGET_COL]
        .mean()
        .sort_values(ascending=False)
        .head(15)
        .reset_index()
    )
    fig_brand = px.bar(
        brand_avg,
        x="brand",
        y=TARGET_COL,
        color=TARGET_COL,
        color_continuous_scale="Blues",
        labels={"brand": "Brand", TARGET_COL: "Average Price ($)"},
    )
    fig_brand.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig_brand, use_container_width=True)
 
with chart_col2:
    st.markdown(f"##### Average Price by Model ({brand if brand and not brand.startswith('--') else 'Select a Brand'})")
    if brand and not brand.startswith("--"):
        model_avg = (
            df[df["brand"] == brand]
            .groupby("model")[TARGET_COL]
            .mean()
            .sort_values(ascending=False)
            .head(15)
            .reset_index()
        )
        fig_model = px.bar(
            model_avg,
            x="model",
            y=TARGET_COL,
            color=TARGET_COL,
            color_continuous_scale="Greens",
            labels={"model": "Model", TARGET_COL: "Average Price ($)"},
        )
        fig_model.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_model, use_container_width=True)
    else:
        st.info("ℹ️ Select a brand above to view its model-level price breakdown.")
 
# ==============================================================================
# SECTION 15: FOOTER
# ==============================================================================
st.markdown("---")
st.markdown(
    """
    <div style="text-align:center; opacity:0.7; padding: 1rem 0;">
        Built with ❤️ using Streamlit & Scikit-Learn — Used Car Price Predictor © 2026
    </div>
    """,
    unsafe_allow_html=True,
)
