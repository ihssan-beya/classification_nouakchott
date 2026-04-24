import streamlit as st
import ee
import folium
from streamlit_folium import st_folium
import backend

st.set_page_config(page_title="Classification LULC Nouakchott", layout="wide")

# =========================
# 🔐 EARTH ENGINE INIT
# =========================
try:
    ee.Initialize(project='non-commercial-471612')
except:
    service_account = st.secrets["GEE_SERVICE_ACCOUNT"]

    credentials = ee.ServiceAccountCredentials(
        service_account,
        "credentials.json"
    )

    ee.Initialize(credentials, project='non-commercial-471612')

# =========================
# 🎨 HEADER
# =========================
st.markdown("""
<div style='background: linear-gradient(135deg, #0f172a, #1e40af); color: white;
     padding: 20px 30px; border-radius: 12px; margin-bottom: 20px;'>
    <h1 style='margin:0;'>Classification LULC - Nouakchott</h1>
    <p style='margin:4px 0 0 0; opacity:0.8;'>Random Forest | Sentinel-2 | Google Earth Engine</p>
</div>
""", unsafe_allow_html=True)

# =========================
# 📊 SIDEBAR
# =========================
with st.sidebar:
    st.markdown("### Parametres")
    st.markdown("Zone : Nouakchott")
    st.markdown("Satellite : Sentinel-2 (2023)")
    st.markdown("Classifieur : Random Forest")
    st.markdown("---")
    st.markdown("0 = Eau")
    st.markdown("1 = Vegetation")
    st.markdown("2 = Urbain")
    st.markdown("3 = Sable")

# =========================
# 📍 ROI
# =========================
roi_coords = [-16.05, 18.05, -15.90, 18.15]

st.markdown("### Zone d'etude")

m = folium.Map(location=[18.10, -15.975], zoom_start=13, tiles="Esri.WorldImagery")

folium.Rectangle(
    bounds=[[roi_coords[1], roi_coords[0]], [roi_coords[3], roi_coords[2]]],
    color="red",
    fill=True,
    fill_opacity=0.1
).add_to(m)

st_folium(m, height=400)

# =========================
# 🚀 RUN CLASSIFICATION
# =========================
if st.button("Lancer la classification", type="primary"):
    with st.spinner("Traitement en cours..."):
        classified = backend.run_analysis(roi_coords)

        vis = {
            'min': 0,
            'max': 3,
            'palette': ['0000FF', '00FF00', 'FF0000', 'FFFF00']
        }

        map_id = classified.getMapId(vis)
        tile_url = map_id["tile_fetcher"].url_format

        st.success("Terminé !")

        m2 = folium.Map(location=[18.10, -15.975], zoom_start=13, tiles="Esri.WorldImagery")

        folium.TileLayer(
            tiles=tile_url,
            attr="GEE",
            overlay=True
        ).add_to(m2)

        st_folium(m2, height=500)
