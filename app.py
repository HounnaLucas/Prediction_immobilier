# ============================================================
# APP STREAMLIT – PRÉDICTION PRIX IMMOBILIER (80 FEATURES)
# ============================================================

import streamlit as st
import pandas as pd
import joblib

# ======================
# CONFIG PAGE
# ======================
st.set_page_config(page_title="Prédiction Prix Immobilier", layout="wide")


st.title("🏠 Prédiction du Prix de l'Immobilier")

# ======================
# FOND CITE + STYLE FORMULAIRE PREMIUM
# ======================
st.markdown(
    """
    <style>
   

    /* Conteneur du formulaire */
    .block-container {
        background-color: rgba(255, 255, 255, 0.95); /* fond blanc semi-transparent pour le bloc */
        padding: 2rem;
        border-radius: 12px;
    }

    /* Titres des cellules */
    .cell-title {
        background-color: #FFA500; /* jaune/orange */
        color: #000000;
        font-weight: bold;
        text-transform: uppercase;
        padding: 2px 6px;
        border-radius: 4px;
        margin-bottom: 2px;
    }

    /* Description sous chaque cellule */
    .cell-help {
        color: #000000; /* noir */
        font-size: 0.85rem;
        margin-top: 2px;
    }

    /* Fond des champs input */
    .stNumberInput, .stSelectbox {
        background-color: #d3d3d3; /* gris clair */
        border-radius: 6px;
        padding: 4px;
    }

    /* Galerie d’images défilable */
    .scrolling-wrapper {
        overflow-x: auto;
        display: flex;
        flex-wrap: nowrap;
        padding-bottom: 1rem;
        margin-bottom: 1rem;
    }
    .scrolling-wrapper .card {
        flex: 0 0 auto;
        width: 220px;
        margin-right: 1rem;
        text-align: center;
    }
    .scrolling-wrapper img {
        border-radius: 10px;
        width: 100%;
        height: 150px;
        object-fit: cover;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ======================
# GALERIE DE MAISONS (12 MAISONS, DÉFILEMENT HORIZONTAL)
# ======================
st.markdown("### Maisons en vogues")

maisons = [
    ("Maison moderne", "https://images.unsplash.com/photo-1568605114967-8130f3a36994"),
    ("Maison familiale", "https://images.unsplash.com/photo-1572120360610-d971b9b78825"),
    ("Villa de luxe", "https://images.unsplash.com/photo-1600585154340-be6161a56a0c"),
    ("Maison urbaine", "https://images.unsplash.com/photo-1598928506311-c55ded91a20c"),
    ("Maison contemporaine", "https://images.unsplash.com/photo-1580587771525-78b9dba3b914"),
    ("Maison en bois", "https://images.unsplash.com/photo-1507089947368-19c1da9775ae"),
    ("Petit pavillon", "https://images.unsplash.com/photo-1618223419713-f96f7c4c99d5"),
    ("Maison de campagne", "https://images.unsplash.com/photo-1600585154276-7ed86b4b3c43"),
    ("Villa moderne", "https://images.unsplash.com/photo-1542317854-7ef12ecf5fa7"),
    ("Maison sur pilotis", "https://images.unsplash.com/photo-1600585154142-4620c6e1ee82"),
    ("Maison en pierre", "https://images.unsplash.com/photo-1568605114968-82e4a3f0468e"),
    ("Maison minimaliste", "https://images.unsplash.com/photo-1600585154255-918987b7f9d1")
]

html = '<div class="scrolling-wrapper">'
for titre, url in maisons:
    html += f'<div class="card"><img src="{url}"><p style="font-weight:600">{titre}</p></div>'
html += '</div>'

st.markdown(html, unsafe_allow_html=True)

# ======================
# FONCTIONS DE FORMULAIRE AVEC STYLE
# ======================
def champ_number(col, key, label, help_txt, default=0):
    col.markdown(f"<div class='cell-title'>{label}</div>", unsafe_allow_html=True)
    val = col.number_input("", value=default, key=key)
    col.markdown(f"<div class='cell-help'>{help_txt}</div>", unsafe_allow_html=True)
    return val

def champ_select(col, key, label, help_txt, options):
    col.markdown(f"<div class='cell-title'>{label}</div>", unsafe_allow_html=True)
    choix = col.selectbox("", list(options.keys()), key=key)
    col.markdown(f"<div class='cell-help'>{help_txt}</div>", unsafe_allow_html=True)
    return options[choix]

st.markdown("Remplissez les caractéristiques de la maison. Toutes les variables du dataset Ames Housing sont incluses.")


# ======================
# CHARGEMENT MODÈLE
# ======================
xgb_model = joblib.load("xgb_model.pkl")
scaler = joblib.load("scaler.pkl")
X_encoded = joblib.load("X_encoded_columns.pkl")

# ======================
# OUTILS UI
# ======================
QUAL = {
    "Excellent": "Ex",
    "Bon": "Gd",
    "Moyen": "TA",
    "Faible": "Fa",
    "Mauvais": "Po",
    "Aucun": "NA"
}

def champ_number(col, key, label, help_txt, default=0):
    col.markdown(f"**{label}**")
    val = col.number_input("", value=default, key=key)
    col.caption(help_txt)
    return val

def champ_select(col, key, label, help_txt, options):
    col.markdown(f"**{label}**")
    choix = col.selectbox(
        "",
        list(options.keys()),
        key=key
    )
    col.caption(help_txt)
    return options[choix]

# ======================
# FORMULAIRE
# ======================
vals = {}

with st.form("form_maison"):

    # ======================
    st.subheader("🏡 1. Structure & Style")
    c = st.columns(3)

    vals["MSSubClass"] = champ_number(c[0], "MSSubClass", "Type maison",
        "Type et époque de construction", 20)

    vals["BldgType"] = champ_select(c[1], "BldgType", "Type logement",
        "Maison individuelle, duplex, etc.",
        {"Maison individuelle": "1Fam", "Duplex": "Duplex", "Townhouse": "Twnhs"})

    vals["HouseStyle"] = champ_select(c[2], "HouseStyle", "Style maison",
        "Style architectural",
        {"1 étage": "1Story", "2 étages": "2Story", "Split": "SLvl"})

    vals["OverallQual"] = champ_number(c[0], "OverallQual", "Qualité globale",
        "Note de qualité (1–10)", 7)

    vals["OverallCond"] = champ_number(c[1], "OverallCond", "État global",
        "Note d’état (1–10)", 5)

    vals["YearBuilt"] = champ_number(c[2], "YearBuilt", "Année construction",
        "Année de construction", 2000)

    vals["YearRemodAdd"] = champ_number(c[0], "YearRemodAdd", "Année rénovation",
        "Dernière rénovation", 2005)

    # ======================
    st.subheader("🌍 2. Terrain")
    c = st.columns(3)

    vals["MSZoning"] = champ_select(c[0], "MSZoning", "Zonage",
        "Type de zone",
        {"Résidentiel faible": "RL", "Résidentiel dense": "RM", "Commercial": "C"})

    vals["LotFrontage"] = champ_number(c[1], "LotFrontage", "Façade",
        "Longueur façade (m)", 80)

    vals["LotArea"] = champ_number(c[2], "LotArea", "Surface terrain",
        "Surface totale du terrain", 9600)

    vals["Street"] = champ_select(c[0], "Street", "Rue",
        "Type de rue",
        {"Pavée": "Pave", "Gravier": "Grvl"})

    vals["Alley"] = champ_select(c[1], "Alley", "Allée",
        "Type d’allée",
        {"Aucune": "NA", "Pavée": "Pave", "Gravier": "Grvl"})

    vals["LotShape"] = champ_select(c[2], "LotShape", "Forme terrain",
        "Forme du terrain",
        {"Régulier": "Reg", "Légèrement irrégulier": "IR1", "Irrégulier": "IR2"})

    vals["LandContour"] = champ_select(c[0], "LandContour", "Contour",
        "Topographie du terrain",
        {"Plat": "Lvl", "Pente douce": "Bnk", "Dépression": "Low"})

    vals["LotConfig"] = champ_select(c[1], "LotConfig", "Configuration",
        "Disposition du terrain",
        {"Intérieur": "Inside", "Angle": "Corner", "Cul-de-sac": "CulDSac"})

    vals["LandSlope"] = champ_select(c[2], "LandSlope", "Pente",
        "Inclinaison du terrain",
        {"Douce": "Gtl", "Modérée": "Mod", "Forte": "Sev"})

    vals["Neighborhood"] = champ_select(c[0], "Neighborhood", "Quartier",
        "Quartier de localisation",
        {"Résidentiel": "CollgCr", "Centre": "NAmes", "Luxe": "NridgHt"})

    vals["Condition1"] = champ_select(c[1], "Condition1", "Condition 1",
        "Influence environnementale principale",
        {"Normale": "Norm", "Route": "Artery", "Rail": "RRAn"})

    vals["Condition2"] = champ_select(c[2], "Condition2", "Condition 2",
        "Influence environnementale secondaire",
        {"Normale": "Norm", "Route": "Artery", "Rail": "RRAn"})

    # ======================
    st.subheader("🏗️ 3. Construction")
    c = st.columns(3)

    vals["RoofStyle"] = champ_select(c[0], "RoofStyle", "Style toit",
        "Forme du toit",
        {"Gable": "Gable", "Hip": "Hip", "Flat": "Flat"})

    vals["RoofMatl"] = champ_select(c[1], "RoofMatl", "Matériau toit",
        "Matériau de couverture",
        {"Bardeaux": "CompShg", "Métal": "Metal", "Tuile": "Tile"})

    vals["Exterior1st"] = champ_select(c[2], "Exterior1st", "Façade principale",
        "Matériau extérieur principal",
        {"Vinyle": "VinylSd", "Brique": "BrkFace", "Bois": "Wd Sdng"})

    vals["Exterior2nd"] = champ_select(c[0], "Exterior2nd", "Façade secondaire",
        "Matériau extérieur secondaire",
        {"Vinyle": "VinylSd", "Brique": "BrkFace", "Bois": "Wd Sdng"})

    vals["MasVnrType"] = champ_select(c[1], "MasVnrType", "Maçonnerie",
        "Type de revêtement maçonné",
        {"Aucun": "None", "Brique": "BrkFace", "Pierre": "Stone"})

    vals["MasVnrArea"] = champ_number(c[2], "MasVnrArea", "Surface maçonnerie",
        "Surface en maçonnerie", 0)

    vals["Foundation"] = champ_select(c[0], "Foundation", "Fondation",
        "Type de fondation",
        {"Béton": "PConc", "Brique": "BrkTil", "Dalle": "Slab"})

    # ======================
    st.subheader("🏠 4. Sous-sol")
    c = st.columns(3)

    vals["BsmtQual"] = champ_select(c[0], "BsmtQual", "Qualité sous-sol",
        "Qualité du sous-sol", QUAL)

    vals["BsmtCond"] = champ_select(c[1], "BsmtCond", "État sous-sol",
        "État du sous-sol", QUAL)

    vals["BsmtExposure"] = champ_select(c[2], "BsmtExposure", "Exposition",
        "Luminosité du sous-sol",
        {"Aucune": "No", "Faible": "Mn", "Bonne": "Gd"})

    vals["BsmtFinType1"] = champ_select(c[0], "BsmtFinType1", "Finition SS1",
        "Type finition principale",
        {"Non fini": "Unf", "Habitable": "GLQ"})

    vals["BsmtFinSF1"] = champ_number(c[1], "BsmtFinSF1", "Surface SS1",
        "Surface finie principale", 0)

    vals["BsmtFinType2"] = champ_select(c[2], "BsmtFinType2", "Finition SS2",
        "Type finition secondaire",
        {"Non fini": "Unf", "Habitable": "ALQ"})

    vals["BsmtFinSF2"] = champ_number(c[0], "BsmtFinSF2", "Surface SS2",
        "Surface finie secondaire", 0)

    vals["BsmtUnfSF"] = champ_number(c[1], "BsmtUnfSF", "Surface non finie",
        "Surface non finie sous-sol", 0)

    vals["TotalBsmtSF"] = champ_number(c[2], "TotalBsmtSF", "Surface totale SS",
        "Surface totale du sous-sol", 0)

    # ======================
    st.subheader("🛋️ 5. Intérieur")
    c = st.columns(3)

    vals["1stFlrSF"] = champ_number(c[0], "1stFlrSF", "Surface RDC",
        "Surface rez-de-chaussée", 900)

    vals["2ndFlrSF"] = champ_number(c[1], "2ndFlrSF", "Surface étage",
        "Surface étage supérieur", 500)

    vals["GrLivArea"] = champ_number(c[2], "GrLivArea", "Surface habitable",
        "Surface habitable totale", 1400)

    vals["BedroomAbvGr"] = champ_number(c[0], "BedroomAbvGr", "Chambres",
        "Nombre de chambres", 3)

    vals["KitchenAbvGr"] = champ_number(c[1], "KitchenAbvGr", "Cuisines",
        "Nombre de cuisines", 1)

    vals["KitchenQual"] = champ_select(c[2], "KitchenQual", "Qualité cuisine",
        "Qualité de la cuisine", QUAL)

    vals["TotRmsAbvGrd"] = champ_number(c[0], "TotRmsAbvGrd", "Pièces totales",
        "Nombre total de pièces", 6)

    vals["Functional"] = champ_select(c[1], "Functional", "Fonctionnalité",
        "Fonctionnalité globale",
        {"Fonctionnelle": "Typ", "Limitée": "Min1"})

    # ======================
    st.subheader("🚗 6. Garage & Extérieur")
    c = st.columns(3)

    vals["GarageType"] = champ_select(c[0], "GarageType", "Type garage",
        "Type de garage",
        {"Attaché": "Attchd", "Détaché": "Detchd", "Aucun": "NA"})

    vals["GarageCars"] = champ_number(c[1], "GarageCars", "Places garage",
        "Capacité du garage", 2)

    vals["GarageArea"] = champ_number(c[2], "GarageArea", "Surface garage",
        "Surface du garage", 400)

    vals["PavedDrive"] = champ_select(c[0], "PavedDrive", "Allée pavée",
        "Allée pavée ou non",
        {"Oui": "Y", "Non": "N"})

    vals["WoodDeckSF"] = champ_number(c[1], "WoodDeckSF", "Terrasse bois",
        "Surface terrasse bois", 0)

    vals["OpenPorchSF"] = champ_number(c[2], "OpenPorchSF", "Porche ouvert",
        "Surface porche ouvert", 0)

    vals["ScreenPorch"] = champ_number(c[0], "ScreenPorch", "Porche écran",
        "Surface porche moustiquaire", 0)

    vals["PoolArea"] = champ_number(c[1], "PoolArea", "Piscine",
        "Surface piscine", 0)

    vals["MiscVal"] = champ_number(c[2], "MiscVal", "Valeur annexe",
        "Valeur équipements annexes", 0)

    # ======================
    st.subheader("💰 7. Vente")
    c = st.columns(3)

    vals["MoSold"] = champ_number(c[0], "MoSold", "Mois vente",
        "Mois de la vente", 6)

    vals["YrSold"] = champ_number(c[1], "YrSold", "Année vente",
        "Année de la vente", 2020)

    vals["SaleType"] = champ_select(c[2], "SaleType", "Type vente",
        "Type de transaction",
        {"Standard": "WD", "Cash": "Cash", "Neuf": "New"})

    vals["SaleCondition"] = champ_select(c[0], "SaleCondition", "Condition vente",
        "Condition de la vente",
        {"Normale": "Normal", "Famille": "Family", "Forclusion": "Abnorml"})

    # ======================
    submit = st.form_submit_button("💰 Prédire le prix")

# ======================
# PRÉDICTION
# ======================
if submit:
    df = pd.DataFrame([vals])
    df_encoded = pd.get_dummies(df)
    df_encoded = df_encoded.reindex(columns=X_encoded, fill_value=0)
    df_scaled = scaler.transform(df_encoded)
    prix = xgb_model.predict(df_scaled)

    st.success(f"🏷️ Prix estimé : **{prix[0]:,.0f} $**")
    st.balloons()
