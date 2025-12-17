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
# Labels français + descriptions
# -------------------
labels_fr = {
    "MSSubClass": ("Type construction", "Classe de construction de la maison"),
    "MSZoning": ("Zone du terrain", "Type de zonage : résidentiel, commercial, etc."),
    "LotFrontage": ("Façade sur rue", "Longueur du terrain le long de la rue en pieds"),
    "LotArea": ("Superficie du terrain", "Surface totale du terrain en pieds²"),
    "Street": ("Type de rue", "Pavé ou non"),
    "Alley": ("Allée", "Type d'accès secondaire ou NA"),
    "LotShape": ("Forme du terrain", "Régulier ou irrégulier"),
    "LandContour": ("Contour du terrain", "Plat ou pente"),
    "Utilities": ("Services publics", "AllPub=Tout disponible, NoSewr=Non"),
    "LotConfig": ("Configuration du lot", "FR2, Inside, Corner, CulDSac"),
    "LandSlope": ("Pente du terrain", "Gtl=Faible, Mod=Moyenne, Sev=Forte"),
    "Neighborhood": ("Quartier", "Nom du quartier"),
    "Condition1": ("Proximité route 1", "Route principale proche de la maison"),
    "Condition2": ("Proximité route 2", "Deuxième route proche de la maison"),
    "BldgType": ("Type de bâtiment", "1Fam=Maison individuelle, 2FmCon=Duplex..."),
    "HouseStyle": ("Style de maison", "1Story, 2Story, etc."),
    "OverallQual": ("Qualité générale", "1=Mauvais, 10=Excellent"),
    "OverallCond": ("État général", "1 à 10"),
    "YearBuilt": ("Année construction", "Année de construction"),
    "YearRemodAdd": ("Année rénovation", "Année de remodelage"),
    "RoofStyle": ("Style toit", "Gable, Hip, Flat..."),
    "RoofMatl": ("Matériau toit", "CompShg, Metal, etc."),
    "Exterior1st": ("Revêtement extérieur 1", "VinylSd, MetalSd, etc."),
    "Exterior2nd": ("Revêtement extérieur 2", "VinylSd, MetalSd, etc."),
    "MasVnrType": ("Type maçonnerie", "None, BrkFace, Stone, etc."),
    "MasVnrArea": ("Surface maçonnerie", "En pieds²"),
    "ExterQual": ("Qualité extérieur", "Ex=Excellent, Gd=Bon, TA=Correct, Fa=Médiocre, Po=Mauvais"),
    "ExterCond": ("État extérieur", "Ex, Gd, TA, Fa, Po"),
    "Foundation": ("Fondation", "PConc, CBlock, BrkTil, Slab, etc."),
    "BsmtQual": ("Qualité sous-sol", "Ex, Gd, TA, Fa, Po, NA"),
    "BsmtCond": ("État sous-sol", "Ex, Gd, TA, Fa, Po, NA"),
    "BsmtExposure": ("Exposition sous-sol", "Gd=Bonne, Av=Moyenne, Mn=Faible, No=Aucune, NA"),
    "BsmtFinType1": ("Type finition 1", "GLQ, ALQ, BLQ, Rec, LwQ, Unf, NA"),
    "BsmtFinSF1": ("Surface finie 1", "En pieds²"),
    "BsmtFinType2": ("Type finition 2", "GLQ, ALQ, BLQ, Rec, LwQ, Unf, NA"),
    "BsmtFinSF2": ("Surface finie 2", "En pieds²"),
    "BsmtUnfSF": ("Sous-sol non fini", "En pieds²"),
    "TotalBsmtSF": ("Surface totale sous-sol", "En pieds²"),
    "1stFlrSF": ("Surface 1er étage", "En pieds²"),
    "2ndFlrSF": ("Surface 2ème étage", "En pieds²"),
    "GrLivArea": ("Surface habitable", "En pieds²"),
    "GarageCars": ("Capacité garage", "Nombre de voitures"),
    "GarageArea": ("Surface garage", "En pieds²"),
    "WoodDeckSF": ("Terrasse bois", "Surface en pieds²"),
    "OpenPorchSF": ("Porche ouvert", "Surface en pieds²"),
    "EnclosedPorch": ("Porche fermé", "Surface en pieds²"),
    "ScreenPorch": ("Porche grillagé", "Surface en pieds²"),
    "PoolArea": ("Piscine", "Surface en pieds²"),
    "MiscVal": ("Valeur divers", "Valeur des commodités diverses"),
    "MoSold": ("Mois de vente", "1=Janvier, 12=Décembre"),
    "YrSold": ("Année de vente", "Ex: 2010, 2015, etc."),
    "Heating": ("Type chauffage", "GasA, GasW, Floor, etc."),
    "HeatingQC": ("Qualité chauffage", "Ex, Gd, TA, Fa, Po"),
    "CentralAir": ("Climatisation centrale", "Y=Oui, N=Non"),
    "Electrical": ("Électricité", "SBrkr, FuseF, FuseA, Mix"),
    "KitchenQual": ("Qualité cuisine", "Ex, Gd, TA, Fa, Po"),
    "Functional": ("Fonctionnalité maison", "Typ=Normal, Min1=Minimale, etc."),
    "FireplaceQu": ("Qualité cheminée", "Ex, Gd, TA, Fa, Po, NA"),
    "GarageType": ("Type garage", "Attchd, Detchd, BuiltIn, CarPort, NA"),
    "GarageFinish": ("Finition garage", "Fin, RFn, Unf, NA"),
    "GarageQual": ("Qualité garage", "Ex, Gd, TA, Fa, Po, NA"),
    "GarageCond": ("État garage", "Ex, Gd, TA, Fa, Po, NA"),
    "PavedDrive": ("Allée pavée", "Y=Oui, P=Partiel, N=Non"),
    "PoolQC": ("Qualité piscine", "Ex, Gd, TA, Fa, Po, NA"),
    "Fence": ("Clôture", "GdPrv, MnPrv, GdWo, MnWw, NA"),
    "MiscFeature": ("Caractéristiques diverses", "Elev, Gar2, Shed, TenC, NA"),
    "SaleType": ("Type de vente", "WD, CWD, VWD, ConLD, ConLI, ConLw, Oth"),
    "SaleCondition": ("Condition vente", "Normal, Abnorml, AdjLand, Alloca, Family, Partial")
}

# -------------------
# Options pour les selectbox
# -------------------
options_dict = {
    "LandContour": {"Lvl":"Plat","Bnk":"Pente","HLS":"Haut-Bas","Low":"Bas"},
    "LotShape": {"Reg":"Régulier","IR1":"Irrégulier 1","IR2":"Irrégulier 2","IR3":"Irrégulier 3"},
    "ExterQual": {"Ex":"Excellent","Gd":"Bon","TA":"Correct","Fa":"Médiocre","Po":"Mauvais"},
    "ExterCond": {"Ex":"Excellent","Gd":"Bon","TA":"Correct","Fa":"Médiocre","Po":"Mauvais"},
    "BsmtQual": {"Ex":"Excellent","Gd":"Bon","TA":"Correct","Fa":"Médiocre","Po":"Mauvais","NA":"Aucun"},
    "BsmtCond": {"Ex":"Excellent","Gd":"Bon","TA":"Correct","Fa":"Médiocre","Po":"Mauvais","NA":"Aucun"},
    "BsmtExposure": {"Gd":"Bonne","Av":"Moyenne","Mn":"Faible","No":"Aucune","NA":"Aucune"},
    "BsmtFinType1": {"GLQ":"Good Living","ALQ":"Average Living","BLQ":"Basement Living","Rec":"Recréation","LwQ":"Low Quality","Unf":"Non fini","NA":"Aucun"},
    "BsmtFinType2": {"GLQ":"Good Living","ALQ":"Average Living","BLQ":"Basement Living","Rec":"Recréation","LwQ":"Low Quality","Unf":"Non fini","NA":"Aucun"},
    "GarageType": {"Attchd":"Attaché","Detchd":"Détaché","BuiltIn":"Intégré","CarPort":"Abri","NA":"Aucun"},
    "GarageFinish": {"Fin":"Fini","RFn":"Semi-fini","Unf":"Non fini","NA":"Aucun"},
    "GarageQual": {"Ex":"Excellent","Gd":"Bon","TA":"Correct","Fa":"Médiocre","Po":"Mauvais","NA":"Aucun"},
    "GarageCond": {"Ex":"Excellent","Gd":"Bon","TA":"Correct","Fa":"Médiocre","Po":"Mauvais","NA":"Aucun"},
    "PavedDrive": {"Y":"Oui","P":"Partiel","N":"Non"},
    "CentralAir": {"Y":"Oui","N":"Non"},
    "FireplaceQu": {"Ex":"Excellent","Gd":"Bon","TA":"Correct","Fa":"Médiocre","Po":"Mauvais","NA":"Aucun"}
}

# -------------------
# Valeurs par défaut
# -------------------
default_values = {field: 0 for field in labels_fr.keys()}
default_values.update({
    "MSSubClass":20, "LotFrontage":80, "LotArea":9600, "OverallQual":7, "OverallCond":5,
    "YearBuilt":2000, "YearRemodAdd":2005, "MasVnrArea":0, "BsmtFinSF1":0, "BsmtFinSF2":0,
    "BsmtUnfSF":0, "TotalBsmtSF":0, "1stFlrSF":900, "2ndFlrSF":500, "GrLivArea":1400,
    "GarageCars":2, "GarageArea":400, "WoodDeckSF":0, "OpenPorchSF":0, "EnclosedPorch":0,
    "ScreenPorch":0, "PoolArea":0, "MiscVal":0, "MoSold":6, "YrSold":2020,
    "MSZoning":"RL", "Street":"Pave", "Alley":"NA", "LotShape":"Reg", "LandContour":"Lvl",
    "Utilities":"AllPub", "LotConfig":"FR2", "LandSlope":"Gtl", "Neighborhood":"CollgCr",
    "Condition1":"Norm", "Condition2":"Norm", "BldgType":"1Fam", "HouseStyle":"2Story",
    "RoofStyle":"Gable", "RoofMatl":"CompShg", "Exterior1st":"VinylSd", "Exterior2nd":"VinylSd",
    "MasVnrType":"None", "ExterQual":"Gd", "ExterCond":"TA", "Foundation":"PConc",
    "BsmtQual":"Gd", "BsmtCond":"TA", "BsmtExposure":"No", "BsmtFinType1":"GLQ", "BsmtFinType2":"Unf",
    "Heating":"GasA", "HeatingQC":"Ex", "CentralAir":"Y", "Electrical":"SBrkr", "KitchenQual":"Gd",
    "Functional":"Typ", "FireplaceQu":"NA", "GarageType":"Attchd", "GarageFinish":"Unf",
    "GarageQual":"TA", "GarageCond":"TA", "PavedDrive":"Y", "PoolQC":"NA", "Fence":"NA",
    "MiscFeature":"NA", "SaleType":"WD", "SaleCondition":"Normal"
})

# -------------------
# Formulaire Streamlit final
# -------------------
st.set_page_config(page_title="Prédiction Prix Immobilier", layout="wide")
st.title("🏠 Prédiction du Prix de l'Immobilier")
st.markdown("Remplissez les informations sur la maison. Les valeurs par défaut sont pré-remplies.")

with st.form(key='maison_form'):
    valeurs = {}
    cols = st.columns(3)
    for i, field in enumerate(labels_fr.keys()):
        col = cols[i % 3]
        label, desc = labels_fr[field]
        default = default_values[field]
        st.markdown(f"**{label}**")  # Label en haut
        col.caption(desc)            # Description en bas

        # Champ catégoriel ou numérique
        if field in options_dict or isinstance(default, str):
            valeurs[field] = col.selectbox(
                label="",
                options=list(options_dict.get(field, {default: default}).keys()),
                format_func=lambda x, f=field: options_dict.get(f, {default: default})[x] if f in options_dict else x,
                index=list(options_dict.get(field, {default: default}).keys()).index(default),
                key=f"{field}_select"  # clé unique
            )
        else:
            valeurs[field] = col.number_input(
                label="",
                value=float(default),
                min_value=0.0,
                key=f"{field}_num"  # clé unique
            )

    submit_button = st.form_submit_button(label="💰 Prédire le prix")

# -------------------
# Prédiction
# -------------------
if submit_button:
    nouvelle_maison_df = pd.DataFrame([valeurs])
    nouvelle_maison_encoded = pd.get_dummies(nouvelle_maison_df)
    nouvelle_maison_encoded = nouvelle_maison_encoded.reindex(columns=X_encoded, fill_value=0)
    nouvelle_maison_scaled = scaler.transform(nouvelle_maison_encoded)
    prix_pred = xgb_model.predict(nouvelle_maison_scaled)
    st.markdown("---")
    st.subheader("💡 Résultat")
    st.success(f"Le prix estimé de cette maison est : **{prix_pred[0]:,.2f} $**")
    st.balloons()
