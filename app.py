import streamlit as st
import ee
import json
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium

st.set_page_config(page_title="Classification LULC Nouakchott", layout="wide")

# --- Initialisation GEE ---
initialized = False

if not initialized:
    try:
        ee.Initialize(project='non-commercial-471612')
        initialized = True
    except:
        pass

if not initialized:
    try:
        creds_dict = dict(st.secrets["gee_credentials"])
        key_data = json.dumps(creds_dict)
        creds = ee.ServiceAccountCredentials(creds_dict["client_email"], key_data=key_data)
        ee.Initialize(creds, project='non-commercial-471612')
        initialized = True
    except Exception as e:
        st.error(f"Erreur GEE : {e}")
        st.stop()

# --- Fonction de classification ---
def run_analysis(roi_coords, training_points):
    roi = ee.Geometry.Rectangle(roi_coords)
    collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(roi).filterDate('2023-01-01', '2023-12-31').filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
    image = collection.median().clip(roi)
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')
    ndbi = image.normalizedDifference(['B11', 'B8']).rename('NDBI')
    input_img = image.select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12']).addBands([ndvi, ndwi, ndbi])

    features = []
    for pt in training_points:
        features.append(ee.Feature(ee.Geometry.Point([pt["lon"], pt["lat"]]), {"class": pt["class"]}))

    classes = ee.FeatureCollection(features)
    samples = input_img.sampleRegions(collection=classes, properties=['class'], scale=10)
    classifier = ee.Classifier.smileRandomForest(50).train(samples, 'class', input_img.bandNames())
    classified = input_img.classify(classifier)
    return classified

# --- Titre ---
st.markdown("""
<div style='background: linear-gradient(135deg, #0f172a, #1e40af); color: white;
     padding: 20px 30px; border-radius: 12px; margin-bottom: 20px;'>
    <h1 style='margin:0;'>Classification LULC - Nouakchott</h1>
    <p style='margin:4px 0 0 0; opacity:0.8;'>Random Forest | Sentinel-2 | Google Earth Engine</p>
</div>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.markdown("### Parametres")
    st.markdown("**Satellite :** Sentinel-2 (2023)")
    st.markdown("**Classifieur :** Random Forest (50 arbres)")
    st.markdown("---")

    st.markdown("### Classes et couleurs")
    st.markdown("""
    | Classe | Type | Couleur carte |
    |--------|------|--------------|
    | 0 | Eau | Bleu |
    | 1 | Vegetation | Vert |
    | 2 | Urbain | Rouge |
    | 3 | Sable | Jaune |
    """)

    st.markdown("---")
    st.markdown("### Mode d'emploi")
    st.markdown("1. Dessinez un rectangle sur la zone")
    st.markdown("2. Placez les points d'entrainement")
    st.markdown("3. Cliquez sur Lancer")

# --- Session state ---
if "training_points" not in st.session_state:
    st.session_state.training_points = []
if "drawn_bounds" not in st.session_state:
    st.session_state.drawn_bounds = None
if "results" not in st.session_state:
    st.session_state.results = None

# --- Etape 1 : Selection de la zone ---
st.markdown("### Etape 1 : Selectionnez la zone d'etude")
st.info("Utilisez l'outil rectangle a gauche de la carte pour dessiner votre zone d'etude.")

m1 = folium.Map(location=[18.10, -15.97], zoom_start=12, tiles="Esri.WorldImagery")

Draw(
    export=False,
    draw_options={
        "polyline": False, "polygon": False, "circle": False,
        "circlemarker": False, "marker": False,
        "rectangle": {"shapeOptions": {"color": "#ff4444", "weight": 3, "fillOpacity": 0.1}},
    },
    edit_options={"edit": False},
).add_to(m1)

map1_data = st_folium(m1, height=400, use_container_width=True, returned_objects=["all_drawings"])

if map1_data and map1_data.get("all_drawings"):
    drawings = map1_data["all_drawings"]
    if len(drawings) > 0:
        geom = drawings[-1].get("geometry", {})
        if geom.get("type") == "Polygon":
            coords = geom["coordinates"][0]
            lons = [c[0] for c in coords]
            lats = [c[1] for c in coords]
            st.session_state.drawn_bounds = [min(lons), min(lats), max(lons), max(lats)]

if st.session_state.drawn_bounds:
    b = st.session_state.drawn_bounds
    st.success(f"Zone selectionnee : Lon [{b[0]:.3f}, {b[2]:.3f}] - Lat [{b[1]:.3f}, {b[3]:.3f}]")
else:
    st.warning("Dessinez un rectangle sur la carte ci-dessus.")

# --- Etape 2 : Points d'entrainement ---
st.markdown("### Etape 2 : Placez les points d'entrainement")
st.info("Ajoutez au moins un point par classe (Eau, Vegetation, Urbain, Sable).")

col1, col2, col3 = st.columns(3)

with col1:
    pt_lat = st.number_input("Latitude", value=18.10, format="%.5f", step=0.001)
with col2:
    pt_lon = st.number_input("Longitude", value=-15.97, format="%.5f", step=0.001)
with col3:
    pt_class = st.selectbox("Classe", options=[
        (0, "Eau"),
        (1, "Vegetation"),
        (2, "Urbain"),
        (3, "Sable"),
    ], format_func=lambda x: x[1])

if st.button("Ajouter ce point"):
    st.session_state.training_points.append({
        "lat": pt_lat,
        "lon": pt_lon,
        "class": pt_class[0],
        "label": pt_class[1],
    })
    st.rerun()

# Afficher les points existants
if st.session_state.training_points:
    st.markdown("**Points d'entrainement :**")

    colors_map = {0: "blue", 1: "green", 2: "red", 3: "orange"}
    emoji_map = {0: "🔵", 1: "🟢", 2: "🔴", 3: "🟡"}

    # Carte avec les points
    center_lat = sum(p["lat"] for p in st.session_state.training_points) / len(st.session_state.training_points)
    center_lon = sum(p["lon"] for p in st.session_state.training_points) / len(st.session_state.training_points)

    m_pts = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles="Esri.WorldImagery")

    if st.session_state.drawn_bounds:
        b = st.session_state.drawn_bounds
        folium.Rectangle(bounds=[[b[1], b[0]], [b[3], b[2]]], color="red", weight=2, fill=False).add_to(m_pts)

    for i, p in enumerate(st.session_state.training_points):
        folium.CircleMarker(
            location=[p["lat"], p["lon"]], radius=8,
            color=colors_map[p["class"]], fill=True, fill_opacity=0.9,
            popup=f"{p['label']} (classe {p['class']})"
        ).add_to(m_pts)

    st_folium(m_pts, height=350, use_container_width=True, returned_objects=[])

    # Liste des points
    for i, p in enumerate(st.session_state.training_points):
        st.markdown(f"{emoji_map[p['class']]} **{p['label']}** — Lat: {p['lat']:.5f}, Lon: {p['lon']:.5f}")

    if st.button("Supprimer tous les points"):
        st.session_state.training_points = []
        st.rerun()

else:
    st.warning("Ajoutez des points d'entrainement (au moins 1 par classe).")

# --- Etape 3 : Lancer ---
st.markdown("### Etape 3 : Lancer la classification")

classes_presentes = set(p["class"] for p in st.session_state.training_points)
classes_manquantes = []
if 0 not in classes_presentes:
    classes_manquantes.append("Eau")
if 1 not in classes_presentes:
    classes_manquantes.append("Vegetation")
if 2 not in classes_presentes:
    classes_manquantes.append("Urbain")
if 3 not in classes_presentes:
    classes_manquantes.append("Sable")

if classes_manquantes:
    st.warning(f"Classes manquantes : {', '.join(classes_manquantes)}")

can_run = st.session_state.drawn_bounds is not None and len(st.session_state.training_points) >= 4 and len(classes_manquantes) == 0

if st.button("Lancer la classification", type="primary", use_container_width=True, disabled=not can_run):
    with st.spinner("Classification en cours (30-60 secondes)..."):
        classified = run_analysis(st.session_state.drawn_bounds, st.session_state.training_points)
        vis_params = {'min': 0, 'max': 3, 'palette': ['0000FF', '00FF00', 'FF0000', 'FFFF00']}
        map_id = classified.getMapId(vis_params)
        tile_url = map_id["tile_fetcher"].url_format
        st.session_state.results = tile_url
    st.rerun()

# --- Resultats ---
if st.session_state.results:
    st.markdown("---")
    st.markdown("### Resultat de la classification")

    b = st.session_state.drawn_bounds
    center_lat = (b[1] + b[3]) / 2
    center_lon = (b[0] + b[2]) / 2

    m2 = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles="Esri.WorldImagery")
    folium.TileLayer(tiles=st.session_state.results, attr="GEE", name="Classification LULC", overlay=True).add_to(m2)

    # Points d'entrainement sur la carte resultat
    colors_map = {0: "blue", 1: "green", 2: "red", 3: "orange"}
    for p in st.session_state.training_points:
        folium.CircleMarker(
            location=[p["lat"], p["lon"]], radius=6,
            color="white", fill=True, fill_color=colors_map[p["class"]], fill_opacity=0.9,
            popup=f"{p['label']}"
        ).add_to(m2)

    folium.LayerControl().add_to(m2)

    # Legende
    legend_html = """
    <div style="position: fixed; bottom: 30px; right: 30px; z-index: 1000;
         background: white; padding: 12px 16px; border-radius: 8px;
         box-shadow: 0 2px 8px rgba(0,0,0,0.3); font-size: 14px;">
        <b>Legende</b><br>
        <span style="background:#0000FF; width:14px; height:14px; display:inline-block; margin-right:6px; border-radius:2px;"></span> Eau<br>
        <span style="background:#00FF00; width:14px; height:14px; display:inline-block; margin-right:6px; border-radius:2px;"></span> Vegetation<br>
        <span style="background:#FF0000; width:14px; height:14px; display:inline-block; margin-right:6px; border-radius:2px;"></span> Urbain<br>
        <span style="background:#FFFF00; width:14px; height:14px; display:inline-block; margin-right:6px; border-radius:2px;"></span> Sable
    </div>
    """
    m2.get_root().html.add_child(folium.Element(legend_html))

    st_folium(m2, height=550, use_container_width=True, returned_objects=[])

    # Legende aussi en dehors de la carte
    st.markdown("#### Legende")
    col1, col2, col3, col4 = st.columns(4)
    col1.markdown("🔵 **Eau**")
    col2.markdown("🟢 **Vegetation**")
    col3.markdown("🔴 **Urbain**")
    col4.markdown("🟡 **Sable**")

    st.markdown(f"**Zone analysee :** Lon [{b[0]:.3f}, {b[2]:.3f}] - Lat [{b[1]:.3f}, {b[3]:.3f}]")
    st.markdown(f"**Points d'entrainement :** {len(st.session_state.training_points)}")
    st.markdown(f"**Classifieur :** Random Forest (50 arbres)")
