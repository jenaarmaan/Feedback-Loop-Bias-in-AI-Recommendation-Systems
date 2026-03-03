import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from simulator import run_simulation, calculate_gini, calculate_diversity

# --- Page Config ---
st.set_page_config(
    page_title="Feedback-Loop Bias Simulator",
    page_icon="🧠",
    layout="wide"
)

# --- Custom Styling ---
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stMetric {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #3e4451;
    }
    .stPlot {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- Title and Intro ---
st.title("🧠 Feedback-Loop Bias in AI Recommendation Systems")
st.markdown("""
This simulator visualizes how recommendation algorithms can create self-reinforcing loops, 
leading to **Popularity Bias** (Winner-take-all) and a **Diversity Drop**.
""")

# --- Sidebar Controls ---
with st.sidebar:
    st.header("⚙️ Simulation Settings")
    
    n_items = st.slider("Number of Items", 50, 500, 200)
    n_users = st.slider("Number of Users", 50, 200, 100)
    cycles = st.slider("Simulation Cycles", 10, 100, 50)
    k_slots = st.number_input("Recommendation Slots (K)", 1, 10, 5)
    
    st.divider()
    
    strength = st.slider("Feedback Loop Strength", 0.0, 1.0, 0.7, help="How much 'exposure' influences user clicks vs intrinsic quality.")
    
    st.divider()
    
    use_mitigation = st.checkbox("Enable Exploration (Phase 4 Mitigation)", value=False)
    epsilon = st.slider("Exploration Rate (Epsilon)", 0.0, 0.5, 0.1, disabled=not use_mitigation)

# --- Run Simulation ---
@st.cache_data
def get_sim_data(_params):
    # Unpack params
    p = _params
    history, items = run_simulation(
        num_items=p['n_items'],
        num_users=p['n_users'],
        cycles=p['cycles'],
        k=p['k_slots'],
        feedback_loop_strength=p['strength'],
        mitigation=p['use_mitigation']
    )
    return history, items

params = {
    'n_items': n_items,
    'n_users': n_users,
    'cycles': cycles,
    'k_slots': k_slots,
    'strength': strength,
    'use_mitigation': use_mitigation
}

if use_mitigation:
    # We update the simulator logic globally or pass epsilon
    # For this dashboard, we'll assume the simulator handles epsilon-greedy if mitigation=True
    pass

with st.spinner("Running Simulation..."):
    history, final_items = run_simulation(
        num_items=n_items,
        num_users=n_users,
        cycles=cycles,
        k=k_slots,
        feedback_loop_strength=strength,
        mitigation=use_mitigation
    )

# --- Process Results ---
ginis = []
diversities = []
for snapshot in history:
    stats = snapshot['item_stats']
    exposures = [s[1] for s in stats]
    ginis.append(calculate_gini(exposures))
    diversities.append(calculate_diversity(stats))

final_exposure = sorted([i.exposure for i in final_items], reverse=True)

# --- Main Dashboard ---
col1, col2 = st.columns(2)

with col1:
    st.metric("Final Gini Coefficient", f"{ginis[-1]:.4f}", 
              delta=f"{ginis[-1] - ginis[0]:.4f}", delta_color="inverse")
    st.help("Gini = 1 means extreme inequality. High Gini = High Bias.")

with col2:
    st.metric("Final Aggregate Diversity", f"{diversities[-1]}", 
              delta=f"{diversities[-1] - diversities[0]}")
    st.help("Number of unique items seen at least once.")

st.divider()

# --- Charts ---
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("📈 Exposure Inequality Over Time")
    df_gini = pd.DataFrame({"Cycle": range(len(ginis)), "Gini": ginis})
    st.line_chart(df_gini.set_index("Cycle"), color="#ff4b4b")

with chart_col2:
    st.subheader("🌐 Diversity Decay Over Time")
    df_div = pd.DataFrame({"Cycle": range(len(diversities)), "Diversity": diversities})
    st.line_chart(df_div.set_index("Cycle"), color="#00d1b2")

st.subheader("📊 Final Distribution of Item Exposure (Log Scale)")
df_exp = pd.DataFrame({"Exposure": final_exposure})
st.bar_chart(df_exp, color="#1c83e1")

st.divider()

# --- Explanation Section ---
st.markdown("""
### 💡 What's happening?
1. **The Loop**: At each cycle, the algorithm recommends the 'best' items (highest clicks).
2. **The Bias**: Users are more likely to click items they see (**Exposure Influence**).
3. **The Result**: 
    - In **Naive Mode**, a few items capture all the traffic, and the rest are 'forgotten' (Gini goes up).
    - In **Mitigation Mode**, random exploration allows niche items to be discovered, keeping diversity high.
""")

if st.button("🔄 Clear Cache & Re-run"):
    st.cache_data.clear()
    st.rerun()
