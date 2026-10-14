import streamlit as st
import ee
import geemap
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Initialize GEE
ee.Initialize(project = 'mygismapping')

# Page Config
st.set_page_config(page_title="Mau Forest Predictive Dashboard", layout="wide")
st.title("🌲 Mau Forest Complex: Predictive Deforestation Dashboard")
st.markdown("*Powered by Google Earth Engine & Random Forest Machine Learning*")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("⚙️ Scenario Controls")
year_slider = st.sidebar.slider("Select Year", 1990, 2060, 2020, step=10)
scenario = st.sidebar.selectbox("Future Scenario (2030-2060)", [
    "Business as Usual (BAU)", 
    "Strict Conservation", 
    "High Development"
])

# Define Scenarios based on probability thresholds
# BAU: Top 15% of pixels at risk are deforested
# Conservation: Top 5% (only extreme risk areas)
# High Dev: Top 30% (accelerated loss)
thresholds = {"Business as Usual (BAU)": 0.15, "Strict Conservation": 0.05, "High Development": 0.30}

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["📊 Statistical Trends", "🗺️ Interactive Geospatial Map", "🧠 ML Model Insights"])

# ==========================================
# TAB 1: STATISTICAL TRENDS
# ==========================================
with tab1:
    st.subheader("Historical Loss & Future Projections")
    
    # Mocking the historical CSV data (In production, read from your actual CSV)
    hist_data = {'Year': [1990, 2000, 2010, 2020], 'Forest_Ha': [275000, 260000, 245000, 230000]}
    df_hist = pd.DataFrame(hist_data)
    
    # Calculate future projections based on the selected scenario threshold
    base_2020 = 230000
    annual_loss_rate = (df_hist.iloc[0]['Forest_Ha'] - df_hist.iloc[-1]['Forest_Ha']) / 30
    
    # Adjust loss rate based on scenario
    if scenario == "Strict Conservation":
        future_mult = 0.3 # Loss drops by 70%
    elif scenario == "High Development":
        future_mult = 1.8 # Loss increases by 80%
    else:
        future_mult = 1.0 # BAU
        
    future_years = [2030, 2040, 2050, 2060]
    future_data = []
    current_area = base_2020
    for yr in future_years:
        current_area -= (annual_loss_rate * 10 * future_mult)
        future_data.append({'Year': yr, 'Forest_Ha': max(current_area, 0)}) # Cannot go below 0
        
    df_future = pd.DataFrame(future_data)
    df_all = pd.concat([df_hist, df_future])
    
    # Plotly Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_hist['Year'], y=df_hist['Forest_Ha'], mode='lines+markers', name='Historical', line=dict(color='green', width=4)))
    fig.add_trace(go.Scatter(x=df_future['Year'], y=df_future['Forest_Ha'], mode='lines+markers', name=f'Predicted ({scenario})', line=dict(color='red', width=4, dash='dash')))
    
    fig.update_layout(title='Mau Forest Cover: 1990 - 2060', yaxis_title='Forest Area (Hectares)', xaxis_title='Year', height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Loss (1990-2020)", f"{df_hist.iloc[0]['Forest_Ha'] - df_hist.iloc[-1]['Forest_Ha']:,} Ha", "-16.3%")
    col2.metric("Projected Loss by 2060", f"{base_2020 - df_future.iloc[-1]['Forest_Ha']:,.0f} Ha", f"-{((base_2020 - df_future.iloc[-1]['Forest_Ha'])/base_2020)*100:.1f}%")
    col3.metric("Remaining Forest (2060)", f"{df_future.iloc[-1]['Forest_Ha']:,.0f} Ha")

# ==========================================
# TAB 2: INTERACTIVE GEOSPATIAL MAP
# ==========================================
with tab2:
    st.subheader("Spatial Distribution of Deforestation")
    st.info("💡 **Tip:** Use the layer control on the right side of the map to toggle between Historical Forest and the ML Predictive Probability map.")
    
    # Initialize geemap
    m = geemap.Map(center=[-0.35, 35.55], zoom=9, height=600)
    m.add_basemap("SATELLITE")
    
    # Add Mau Boundary
    m.add_layer(ee.FeatureCollection(ee.Geometry.Polygon([
        [[35.20, -0.80], [35.90, -0.80], [35.95, -0.40], 
         [35.85, 0.10], [35.50, 0.20], [35.25, 0.05], 
         [35.15, -0.40], [35.20, -0.80]]
    ])), {}, "Mau Forest Boundary")
    
    # Add Historical Forest Mask (2020)
    # (Re-using the function logic from the engine script for the map)
    def get_landsat_forest_mask_map(year):
        col = 'LANDSAT/LC08/C02/T1_L2'
        img = ee.ImageCollection(col).filterDate(f'{year}-06-01', f'{year}-09-30').median()
        ndvi = img.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')
        return ndvi.gt(0.45).selfMask()
        
    hist_forest = get_landsat_forest_mask_map(2020)
    m.add_layer(hist_forest, {'palette': ['006400']}, "2020 Forest Cover", True)
    
    # Add Predictive Probability Map
    # Note: In a real app, load the exported GeoTIFF from Google Drive here. 
    # For demonstration, we will generate a dummy probability layer or load from an asset.
    # Assuming you uploaded the exported image as an Earth Engine Asset named 'users/your_username/Mau_2030_Probability'
    try:
        # REPLACE 'users/your_username/Mau_2030_Probability' with your actual asset ID
        prob_image = ee.Image('users/your_username/Mau_2030_Probability') 
        
        # Apply scenario threshold to create a binary risk map
        threshold = thresholds[scenario]
        risk_map = prob_image.gt(threshold).selfMask()
        
        m.add_layer(prob_image, {'min': 0, 'max': 1, 'palette': ['blue', 'yellow', 'red']}, "Deforestation Probability (ML)", False)
        m.add_layer(risk_map, {'palette': ['ff0000']}, f"Predicted Loss: {scenario}", True)
        
    except Exception as e:
        st.warning(f"Predictive map asset not found. Please run `predictive_engine.py` and export the image to GEE Assets. Error: {e}")
        # Fallback dummy layer
        m.add_layer(ee.Image(0).selfMask(), {}, "Predicted Loss (Run Engine First)", True)

    # Render map
    m.to_streamlit(height=600)

# ==========================================
# TAB 3: ML MODEL INSIGHTS
# ==========================================
with tab3:
    st.subheader("Machine Learning Feature Importance")
    st.markdown("Which factors are driving deforestation in the Mau Forest? The Random Forest model ranks the importance of each variable.")
    
    # Mock feature importance data (Replace with actual model output if extracted)
    features = ['Distance to Forest Edge', 'Human Pressure (Nightlights)', 'Slope', 'Elevation']
    importance = [0.45, 0.30, 0.15, 0.10]
    
    df_imp = pd.DataFrame({'Feature': features, 'Importance': importance})
    
    fig_imp = px.bar(df_imp, x='Importance', y='Feature', orientation='h', 
                     title='Feature Importance in Deforestation Prediction',
                     color='Importance', color_continuous_scale='Reds')
    st.plotly_chart(fig_imp, use_container_width=True)
    
    st.markdown("""
    **Model Interpretation:**
    *   **Distance to Edge:** Deforestation is highly concentrated at the boundaries of the forest (edge effect).
    *   **Human Pressure:** Areas with higher nighttime lights (settlements/agriculture) show a strong correlation with forest loss.
    *   **Topography:** Steeper slopes and higher elevations are slightly less prone to deforestation due to physical inaccessibility.
    """)
