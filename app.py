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
    st.markdown("### Mode d'emploi")
    st.markdown("1. Dessinez un rectangle")
    st.markdown("2. Choisissez une classe")
    st.markdown("3. Cliquez sur la carte pour ajouter un point")
    st.markdown("4. Repetez pour chaque classe")
    st.markdown("5. Lancez la classification")
    st.markdown("---")
    st.markdown("### Legende des classes")
    st.markdown("🔵 **0 = Eau**")
    st.markdown("🟢 **1 = Vegetation**")
    st.markdown("🔴 **2 = Urbain**")
    st.markdown("🟡 **3 = Sable**")

# --- Session state ---
if "training_points" not in st.session_state:
    st.session_state.training_points = []
if "drawn_bounds" not in st.session_state:
    st.session_state.drawn_bounds = None
if "results" not in st.session_state:
    st.session_state.results = None

# --- Couleurs ---
colors_map = {0: "blue", 1: "green", 2: "red", 3: "orange"}
emoji_map = {0: "🔵 Eau", 1: "🟢 Vegetation", 2: "🔴 Urbain", 3: "🟡 Sable"}

# --- Classe a ajouter ---
st.markdown("### Etape 1 : Dessinez la zone + placez les points")

col_class, col_info = st.columns([1, 2])

with col_class:
    selected_class = st.radio(
        "Classe du prochain clic :",
        options=[0, 1, 2, 3],
        format_func=lambda x: emoji_map[x],
        horizontal=True,
    )

with col_info:
    st.info("Dessinez un rectangle, puis cliquez sur la carte pour ajouter des points de la classe selectionnee.")

# --- Carte principale ---
m = folium.Map(location=[18.10, -15.97], zoom_start=12, tiles="Esri.WorldImagery")

# Outil de dessin rectangle
Draw(
    export=False,
    draw_options={
        "polyline": False, "polygon": False, "circle": False,
        "circlemarker": False, "marker": False,
        "rectangle": {"shapeOptions": {"color": "#ff4444", "weight": 3, "fillOpacity": 0.1}},
    },
    edit_options={"edit": False},
).add_to(m)

# Afficher la zone selectionnee
if st.session_state.drawn_bounds:
    b = st.session_state.drawn_bounds
    folium.Rectangle(bounds=[[b[1], b[0]], [b[3], b[2]]], color="red", weight=2, fill=False, dash_array="5").add_to(m)

# Afficher les points existants
for p in st.session_state.training_points:
    folium.CircleMarker(
        location=[p["lat"], p["lon"]], radius=8,
        color="white", weight=2, fill=True,
        fill_color=colors_map[p["class"]], fill_opacity=0.9,
        popup=f"{emoji_map[p['class']]}",
    ).add_to(m)

map_data = st_folium(m, height=500, use_container_width=True, returned_objects=["all_drawings", "last_clicked"])

# --- Capturer le rectangle ---
if map_data and map_data.get("all_drawings"):
    drawings = map_data["all_drawings"]
    if len(drawings) > 0:
        geom = drawings[-1].get("geometry", {})
        if geom.get("type") == "Polygon":
            coords = geom["coordinates"][0]
            lons = [c[0] for c in coords]
            lats = [c[1] for c in coords]
            new_bounds = [min(lons), min(lats), max(lons), max(lats)]
            if new_bounds != st.session_state.drawn_bounds:
                st.session_state.drawn_bounds = new_bounds
                st.rerun()

# --- Capturer le clic pour ajouter un point ---
if map_data and map_data.get("last_clicked"):
    clicked = map_data["last_clicked"]
    click_lat = clicked["lat"]
    click_lon = clicked["lng"]

    already_exists = any(
        abs(p["lat"] - click_lat) < 0.0001 and abs(p["lon"] - click_lon) < 0.0001
        for p in st.session_state.training_points
    )

    if not already_exists:
        st.session_state.training_points.append({
            "lat": click_lat,
            "lon": click_lon,
            "class": selected_class,
        })
        st.rerun()

# --- Afficher info zone ---
if st.session_state.drawn_bounds:
    b = st.session_state.drawn_bounds
    st.success(f"Zone selectionnee : Lon [{b[0]:.3f}, {b[2]:.3f}] - Lat [{b[1]:.3f}, {b[3]:.3f}]")

# --- Afficher les points ---
if st.session_state.training_points:
    st.markdown("**Points d'entrainement :**")
    cols = st.columns(4)
    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for p in st.session_state.training_points:
        counts[p["class"]] += 1
    cols[0].metric("🔵 Eau", counts[0])
    cols[1].metric("🟢 Vegetation", counts[1])
    cols[2].metric("🔴 Urbain", counts[2])
    cols[3].metric("🟡 Sable", counts[3])

    if st.button("Supprimer tous les points"):
        st.session_state.training_points = []
        st.session_state.results = None
        st.rerun()

    if st.button("Supprimer le dernier point"):
        st.session_state.training_points.pop()
        st.rerun()

# --- Verification des classes ---
classes_presentes = set(p["class"] for p in st.session_state.training_points)
classes_manquantes = []
for c, name in [(0, "Eau"), (1, "Vegetation"), (2, "Urbain"), (3, "Sable")]:
    if c not in classes_presentes:
        classes_manquantes.append(name)

if classes_manquantes and len(st.session_state.training_points) > 0:
    st.warning(f"Classes manquantes : {', '.join(classes_manquantes)}")

# --- Lancer ---
st.markdown("### Etape 2 : Lancer la classification")

can_run = (
    st.session_state.drawn_bounds is not None
    and len(st.session_state.training_points) >= 4
    and len(classes_manquantes) == 0
)

if st.button("Lancer la classification", type="primary", use_container_width=True, disabled=not can_run):
    with st.spinner("Classification en cours (30-60 secondes)..."):
        classified = run_analysis(st.session_state.drawn_bounds, st.session_state.training_points)
        vis_params = {'min': 0, 'max': 3, 'palette': ['0000FF', '00FF00', 'FF0000', 'FFFF00']}
        map_id = classified.getMapId(vis_params)
        st.session_state.results = map_id["tile_fetcher"].url_format
    st.rerun()

# --- Resultats ---
if st.session_state.results:
    st.markdown("---")
    st.markdown("### Resultat de la classification")

    b = st.session_state.drawn_bounds
    center_lat = (b[1] + b[3]) / 2
    center_lon = (b[0] + b[2]) / 2

    m2 = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles="Esri.WorldImagery")
    folium.TileLayer(tiles=st.session_state.results, attr="GEE", name="Classification", overlay=True).add_to(m2)

    for p in st.session_state.training_points:
        folium.CircleMarker(
            location=[p["lat"], p["lon"]], radius=6,
            color="white", weight=2, fill=True,
            fill_color=colors_map[p["class"]], fill_opacity=0.9,
        ).add_to(m2)

    folium.LayerControl().add_to(m2)

    legend_html = """
    <div style="position: fixed; bottom: 30px; right: 30px; z-index: 1000;
         background: white; padding: 12px 16px; border-radius: 8px;
         box-shadow: 0 2px 8px rgba(0,0,0,0.3); font-size: 14px;">
        <b>Legende</b><br>
        <span style="background:#0000FF; width:14px; height:14px; display:inline-block; margin-right:6px;"></span> Eau<br>
        <span style="background:#00FF00; width:14px; height:14px; display:inline-block; margin-right:6px;"></span> Vegetation<br>
        <span style="background:#FF0000; width:14px; height:14px; display:inline-block; margin-right:6px;"></span> Urbain<br>
        <span style="background:#FFFF00; width:14px; height:14px; display:inline-block; margin-right:6px;"></span> Sable
    </div>
    """
    m2.get_root().html.add_child(folium.Element(legend_html))

    st_folium(m2, height=550, use_container_width=True, returned_objects=[])

    col1, col2, col3, col4 = st.columns(4)
    col1.markdown("🔵 **Eau**")
    col2.markdown("🟢 **Vegetation**")
    col3.markdown("🔴 **Urbain**")
    col4.markdown("🟡 **Sable**")

    st.markdown(f"**Points d'entrainement :** {len(st.session_state.training_points)}")
