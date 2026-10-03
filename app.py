import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

# ---------------------------------------------------------
# PAGE CONFIGURATION & STYLING
# ---------------------------------------------------------
st.set_page_config(
    page_title="Supercapacitor Electrochemical Workbench",
    page_icon="🔬",
    layout="wide"
)

st.markdown("""
    <style>
    .main-header { font-size: 26px; font-weight: 700; color: #1E293B; margin-bottom: 0px; }
    .sub-header { font-size: 14px; color: #64748B; margin-bottom: 25px; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">Supercapacitor Electrochemical Data Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Automated GCD Extraction, Numerical Energy Integration & LOOCV Machine Learning Pipeline</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# SIDEBAR CONTROL PANEL
# ---------------------------------------------------------
st.sidebar.header("Experimental Parameters")
mass_loading = st.sidebar.number_input("Mass Loading (mg/cm²)", value=1.0, step=0.1)
cell_type = st.sidebar.radio("Cell Configuration", ["3-Electrode System", "2-Electrode Symmetric"])

st.sidebar.divider()
uploaded_file = st.sidebar.file_uploader("Upload Raw GCD Excel File (.xlsx)", type=["xlsx"])

# ---------------------------------------------------------
# CALCULATION & VISUALIZATION ENGINE
# ---------------------------------------------------------
def process_gcd_data(file):
    df = pd.read_excel(file)
    time_series = pd.to_numeric(df['Time (s)'].iloc[2:], errors='coerce')
    
    cd_names = ['1 A/g', '2 A/g', '3 A/g', '4 A/g', '5 A/g']
    cols = ['Potential', 'Unnamed: 8', 'Unnamed: 9', 'Unnamed: 10', 'Unnamed: 11']
    
    calc_data = []
    
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    
    for name, col in zip(cd_names, cols):
        if col not in df.columns:
            continue
        pot = pd.to_numeric(df[col].iloc[2:], errors='coerce')
        valid = pd.DataFrame({'Time': time_series, 'Potential': pot}).dropna().reset_index(drop=True)
        
        if len(valid) == 0:
            continue
            
        # Peak detection for discharge region
        max_idx = valid['Potential'].idxmax()
        discharge_df = valid.iloc[max_idx:].reset_index(drop=True)
        
        dt = discharge_df.iloc[-1]['Time'] - discharge_df.loc[0, 'Time']
        dv = abs(discharge_df.loc[0, 'Potential'] - discharge_df.iloc[-1]['Potential'])
        cd_val = float(name.split()[0])
        
        # Specific Capacitance (F/g)
        sc = (cd_val * dt) / dv
        if cell_type == "2-Electrode Symmetric":
            sc *= 4
            
        # Specific Energy via Trapezoidal Integration (Wh/kg)
        # Compatible with both NumPy 1.x (np.trapz) and NumPy 2.x (np.trapezoid)
        v_pos = np.abs(discharge_df['Potential'].values - discharge_df.iloc[-1]['Potential'])
        t_vals = discharge_df['Time'].values
        
        trapz_fn = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
        integral_v_dt = trapz_fn(v_pos, t_vals)
        se = (cd_val * integral_v_dt) / 3.6
        
        # Plotting discharge curve
        ax.plot(discharge_df['Time'] - discharge_df.loc[0, 'Time'], discharge_df['Potential'], label=name, linewidth=1.5)
        
        calc_data.append({
            'Current Density (A/g)': cd_val,
            'Discharge Time (s)': round(dt, 2),
            'Voltage Window (V)': round(dv, 4),
            'Specific Capacitance (F/g)': round(sc, 2),
            'Specific Energy (Wh/kg)': round(se, 2)
        })
        
    ax.set_xlabel("Discharge Duration (s)", fontsize=10)
    ax.set_ylabel("Potential (V)", fontsize=10)
    ax.set_title("Galvanostatic Discharge Curves", fontsize=11, fontweight='bold')
    ax.legend(frameon=True, fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    
    return pd.DataFrame(calc_data), fig

# ---------------------------------------------------------
# MAIN WORKFLOW DISPLAY
# ---------------------------------------------------------
if uploaded_file:
    tab1, tab2, tab3 = st.tabs(["📊 Electrochemical Analysis", "🤖 ML Model Validation", "📖 Governing Equations"])
    
    res_df, gcd_fig = process_gcd_data(uploaded_file)
    
    with tab1:
        st.subheader("Automated Metric Extraction")
        col_left, col_right = st.columns([1.1, 0.9])
        
        with col_left:
            st.dataframe(res_df, use_container_width=True, hide_index=True)
            
            # Key highlight metrics
            sc_1a = res_df.loc[res_df['Current Density (A/g)'] == 1.0, 'Specific Capacitance (F/g)'].values[0]
            se_1a = res_df.loc[res_df['Current Density (A/g)'] == 1.0, 'Specific Energy (Wh/kg)'].values[0]
            
            m1, m2 = st.columns(2)
            m1.metric("Max SC (at 1 A/g)", f"{sc_1a} F/g")
            m2.metric("Max Energy Density", f"{se_1a} Wh/kg")
            
        with col_right:
            st.pyplot(gcd_fig)
            
    with tab2:
        st.subheader("Leave-One-Out Cross Validation (LOOCV)")
        st.markdown("Predicting Electrochemical Performance across Synthesis Parameters ($N=4$ Samples)")
        
        sample_df = pd.DataFrame({
            'Sample ID': ['NM 1', 'NM 2', 'NM 3', 'NM 4'],
            'Decomposition Time (min)': [30, 60, 90, 120],
            'SC at 1 A/g (F/g)': [420.0, 510.0, sc_1a, 480.0]
        })
        
        edited_df = st.data_editor(sample_df, use_container_width=True, hide_index=True)
        
        if st.button("Train and Evaluate ML Pipeline"):
            X = edited_df[['Decomposition Time (min)']]
            y = edited_df['SC at 1 A/g (F/g)']
            
            models = {
                "Random Forest Regressor": RandomForestRegressor(n_estimators=25, random_state=42),
                "Support Vector Machine (SVR)": SVR(kernel='rbf', C=10.0),
                "Artificial Neural Network (MLP)": MLPRegressor(hidden_layer_sizes=(8, 4), max_iter=2000, random_state=42),
                "XGBoost Regressor": XGBRegressor(n_estimators=15, max_depth=2, learning_rate=0.05, random_state=42)
            }
            
            loo = LeaveOneOut()
            eval_results = []
            
            for name, model in models.items():
                y_true, y_pred = [], []
                for train_idx, test_idx in loo.split(X):
                    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
                    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
                    
                    model.fit(X_tr, y_tr)
                    y_pred.append(model.predict(X_te)[0])
                    y_true.append(y_te.values[0])
                    
                mae = np.mean(np.abs(np.array(y_true) - np.array(y_pred)))
                eval_results.append({"Algorithm": name, "LOOCV MAE (F/g)": round(mae, 2)})
                
            st.dataframe(pd.DataFrame(eval_results), use_container_width=True, hide_index=True)
            
    with tab3:
        st.subheader("Mathematical Formulations")
        st.write("Specific Capacitance ($SC$):")
        st.latex(r"SC = \frac{I \cdot \Delta t}{m \cdot \Delta V}")
        
        st.write("Specific Energy via Discharge Integration ($SE$):")
        st.latex(r"SE = \frac{I}{3.6 \cdot m} \int_{t_0}^{t_{\text{end}}} V(t) \, dt")
else:
    st.info("👈 Upload the `NM 3 GCD.xlsx` file from the sidebar to begin analysis.")