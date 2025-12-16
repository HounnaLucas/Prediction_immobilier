import streamlit as st
import pandas as pd
import joblib

# -------------------
# Charger le modèle et le scaler
# -------------------
xgb_model = joblib.load("xgb_model.pkl")
scaler = joblib.load("scaler.pkl")
X_encoded = joblib.load("X_encoded_columns.pkl")  # colonnes après one-hot encoding

# -------------------
# Page et titre
# -------------------
st.set_page_config(page_title="Prédiction Prix Immobilier", layout="wide")
st.title("🏠 Prédiction du Prix de l'Immobilier")
st.markdown("""
Remplissez les informations sur la maison.  
Les valeurs par défaut sont pré-remplies pour faciliter l'utilisation.
""")

# -------------------
# Valeurs par défaut et options
# -------------------
default_values = {
    # Exemple de champs numériques
    "MSSubClass": 20,
    "LotFrontage": 80,
    "LotArea": 9600,
    "OverallQual": 7,
    "OverallCond": 5,
    "YearBuilt": 2000,
    "YearRemodAdd": 2005,
    "MasVnrArea": 0,
    "BsmtFinSF1": 0,
    "BsmtFinSF2": 0,
    "BsmtUnfSF": 0,
    "TotalBsmtSF": 0,
    "1stFlrSF": 900,
    "2ndFlrSF": 500,
    "GrLivArea": 1400,
    "GarageCars": 2,
    "GarageArea": 400,
    "WoodDeckSF": 0,
    "OpenPorchSF": 0,
    "EnclosedPorch": 0,
    "ScreenPorch": 0,
    "PoolArea": 0,
    "MiscVal": 0,
    "MoSold": 6,
    "YrSold": 2020,
    # Exemple de champs catégoriels
    "MSZoning": "RL",
    "Street": "Pave",
    "Alley": "NA",
    "LotShape": "Reg",
    "LandContour": "Lvl",
    "Utilities": "AllPub",
    "LotConfig": "FR2",
    "LandSlope": "Gtl",
    "Neighborhood": "CollgCr",
    "Condition1": "Norm",
    "Condition2": "Norm",
    "BldgType": "1Fam",
    "HouseStyle": "2Story",
    "RoofStyle": "Gable",
    "RoofMatl": "CompShg",
    "Exterior1st": "VinylSd",
    "Exterior2nd": "VinylSd",
    "MasVnrType": "None",
    "ExterQual": "Gd",
    "ExterCond": "TA",
    "Foundation": "PConc",
    "BsmtQual": "Gd",
    "BsmtCond": "TA",
    "BsmtExposure": "No",
    "BsmtFinType1": "GLQ",
    "BsmtFinType2": "Unf",
    "Heating": "GasA",
    "HeatingQC": "Ex",
    "CentralAir": "Y",
    "Electrical": "SBrkr",
    "KitchenQual": "Gd",
    "Functional": "Typ",
    "FireplaceQu": "NA",
    "GarageType": "Attchd",
    "GarageFinish": "Unf",
    "GarageQual": "TA",
    "GarageCond": "TA",
    "PavedDrive": "Y",
    "PoolQC": "NA",
    "Fence": "NA",
    "MiscFeature": "NA",
    "SaleType": "WD",
    "SaleCondition": "Normal"
}

# -------------------
# Créer le formulaire en sections
# -------------------
with st.form(key='maison_form'):
    st.subheader("🏡 Informations sur le terrain")
    cols = st.columns(3)
    terrain_fields = ["MSSubClass", "MSZoning", "LotFrontage", "LotArea", "Street", "Alley", "LotShape", "LandContour", "Utilities", "LotConfig", "LandSlope", "Neighborhood", "Condition1", "Condition2"]
    terrain_vals = {}
    for i, field in enumerate(terrain_fields):
        col = cols[i % 3]
        default = default_values[field]
        if isinstance(default, (int, float)):
            terrain_vals[field] = col.number_input(field, value=default)
        else:
            terrain_vals[field] = col.selectbox(field, [default], index=0)  # tu peux mettre toutes les options ici

    st.subheader("🏠 Informations sur la maison")
    cols = st.columns(3)
    maison_fields = ["BldgType", "HouseStyle", "OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd", "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd", "MasVnrType", "MasVnrArea", "ExterQual", "ExterCond", "Foundation"]
    maison_vals = {}
    for i, field in enumerate(maison_fields):
        col = cols[i % 3]
        default = default_values[field]
        if isinstance(default, (int, float)):
            maison_vals[field] = col.number_input(field, value=default)
        else:
            maison_vals[field] = col.selectbox(field, [default], index=0)

    st.subheader("🏢 Informations sur le sous-sol")
    cols = st.columns(3)
    bsmt_fields = ["BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinSF1", "BsmtFinType2", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF"]
    bsmt_vals = {}
    for i, field in enumerate(bsmt_fields):
        col = cols[i % 3]
        default = default_values[field]
        if isinstance(default, (int, float)):
            bsmt_vals[field] = col.number_input(field, value=default)
        else:
            bsmt_vals[field] = col.selectbox(field, [default], index=0)

    st.subheader("🏡 Informations générales")
    cols = st.columns(3)
    general_fields = ["1stFlrSF", "2ndFlrSF", "GrLivArea", "GarageCars", "GarageArea", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "ScreenPorch", "PoolArea", "MiscVal", "MoSold", "YrSold"]
    general_vals = {}
    for i, field in enumerate(general_fields):
        col = cols[i % 3]
        general_vals[field] = col.number_input(field, value=default_values[field])

    st.subheader("🏘️ Garage, Chauffage, Cuisine et autres")
    cols = st.columns(3)
    other_fields = ["Heating", "HeatingQC", "CentralAir", "Electrical", "KitchenQual", "Functional", "FireplaceQu", "GarageType", "GarageFinish", "GarageQual", "GarageCond", "PavedDrive", "PoolQC", "Fence", "MiscFeature", "SaleType", "SaleCondition"]
    other_vals = {}
    for i, field in enumerate(other_fields):
        col = cols[i % 3]
        default = default_values[field]
        other_vals[field] = col.selectbox(field, [default], index=0)

    # Fusionner toutes les valeurs
    valeurs_maison = {**terrain_vals, **maison_vals, **bsmt_vals, **general_vals, **other_vals}

    submit_button = st.form_submit_button(label='💰 Prédire le prix')

# -------------------
# Prédiction
# -------------------
if submit_button:
    nouvelle_maison_df = pd.DataFrame([valeurs_maison])

    # Encodage identique à X_train
    nouvelle_maison_encoded = pd.get_dummies(nouvelle_maison_df)
    # X_encoded_columns est un Index, pas un DataFrame
    nouvelle_maison_encoded = nouvelle_maison_encoded.reindex(columns=X_encoded, fill_value=0)


    # Standardisation
    nouvelle_maison_scaled = scaler.transform(nouvelle_maison_encoded)

    # Prédiction
    prix_pred = xgb_model.predict(nouvelle_maison_scaled)

    # Affichage du résultat
    st.markdown("---")
    st.subheader("💡 Résultat")
    st.success(f"Le prix estimé de cette maison est : **{prix_pred[0]:,.2f} $**")
    st.balloons()
