import streamlit as st
import ee
import json
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="Classification LULC Nouakchott", layout="wide")

try:
    ee.Initialize(project='non-commercial-471612')
except:
    try:
        creds_dict = dict(st.secrets["gee_credentials"])
        creds = ee.ServiceAccountCredentials(creds_dict["client_email"], key_data=json.dumps(creds_dict))
        ee.Initialize(creds, project='non-commercial-471612')
    except:
        ee.Authenticate()
        ee.Initialize(project='non-commercial-471612')

def run_analysis(roi_coords):
    roi = ee.Geometry.Rectangle(roi_coords)
    collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(roi).filterDate('2023-01-01', '2023-12-31').filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
    image = collection.median().clip(roi)
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')
    ndbi = image.normalizedDifference(['B11', 'B8']).rename('NDBI')
    input_img = image.select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12']).addBands([ndvi, ndwi, ndbi])
    eau = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([-15.98, 18.10]), {'class': 0})])
    vegetation = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([-15.95, 18.08]), {'class': 1})])
    urbain = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([-15.97, 18.09]), {'class': 2})])
    sable = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([-16.00, 18.12]), {'class': 3})])
    classes = eau.merge(vegetation).merge(urbain).merge(sable)
    samples = input_img.sampleRegions(collection=classes, properties=['class'], scale=10)
    classifier = ee.Classifier.smileRandomForest(50).train(samples, 'class', input_img.bandNames())
    classified = input_img.classify(classifier)
    return classified

st.markdown("""
<div style='background: linear-gradient(135deg, #0f172a, #1e40af); color: white;
     padding: 20px 30px; border-radius: 12px; margin-bottom: 20px;'>
    <h1 style='margin:0;'>Classification LULC - Nouakchott</h1>
    <p style='margin:4px 0 0 0; opacity:0.8;'>Random Forest | Sentinel-2 | Google Earth Engine</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Parametres")
    st.markdown("**Zone :** Nouakchott")
    st.markdown("**Satellite :** Sentinel-2 (2023)")
    st.markdown("**Classifieur :** Random Forest (50 arbres)")
    st.markdown("---")
    st.markdown("### Classes")
    st.markdown("0 = Eau")
    st.markdown("1 = Vegetation")
    st.markdown("2 = Urbain")
    st.markdown("3 = Sable")

roi_coords = [-16.05, 18.05, -15.90, 18.15]

st.markdown("### Zone d'etude et points d'entrainement")
m = folium.Map(location=[18.10, -15.975], zoom_start=13, tiles="Esri.WorldImagery")
folium.Rectangle(
    bounds=[[roi_coords[1], roi_coords[0]], [roi_coords[3], roi_coords[2]]],
    color="red", weight=2, fill=True, fill_opacity=0.1
).add_to(m)

points = [
    {"lat": 18.10, "lon": -15.98, "label": "Eau", "color": "blue"},
    {"lat": 18.08, "lon": -15.95, "label": "Vegetation", "color": "green"},
    {"lat": 18.09, "lon": -15.97, "label": "Urbain", "color": "red"},
    {"lat": 18.12, "lon": -16.00, "label": "Sable", "color": "orange"},
]
for p in points:
    folium.CircleMarker(
        location=[p["lat"], p["lon"]], radius=6,
        color=p["color"], fill=True, fill_opacity=0.9,
        popup=p["label"]
    ).add_to(m)

st_folium(m, height=400, use_container_width=True, returned_objects=[])

if st.button("Lancer la classification", type="primary", use_container_width=True):
    with st.spinner("Classification en cours..."):
        classified = run_analysis(roi_coords)
        vis_params = {'min': 0, 'max': 3, 'palette': ['0000FF', '00FF00', 'FF0000', 'FFFF00']}
        map_id = classified.getMapId(vis_params)
        tile_url = map_id["tile_fetcher"].url_format
        st.success("Classification terminee !")
        st.markdown("### Resultat de la classification")
        m2 = folium.Map(location=[18.10, -15.975], zoom_start=13, tiles="Esri.WorldImagery")
        folium.TileLayer(tiles=tile_url, attr="GEE", name="Classification LULC", overlay=True).add_to(m2)
        folium.LayerControl().add_to(m2)
        st_folium(m2, height=500, use_container_width=True, returned_objects=[])
        col1, col2, col3, col4 = st.columns(4)
        col1.markdown("Eau (bleu)")
        col2.markdown("Vegetation (vert)")
        col3.markdown("Urbain (rouge)")
        col4.markdown("Sable (jaune)")
