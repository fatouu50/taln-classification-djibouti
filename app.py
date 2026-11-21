# app.py
# Tableau de bord de démonstration du projet TALN :
# chiffres clés, test d'un titre en direct, répartition du corpus, comparaison des deux approches.
# Lancer depuis le dossier du projet :  streamlit run app.py

import base64

import altair as alt
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Classification thématique", layout="wide")

# ---------------------------------------------------------------- 1. Couleurs (nuances de bleu, comme le modèle de dashboard)
MARINE = "#0B2E59"
BLEU = "#1F5FA8"
BLEU_MOYEN = "#5B8FD1"
BLEU_CLAIR = "#A9C6EB"
BLEU_PALE = "#E8F0FA"
GRIS_TEXTE = "#5A6472"
PALETTE = [MARINE, BLEU, BLEU_MOYEN, "#7FA9DD", BLEU_CLAIR, "#D3E2F5"]

NOMS = {"politique": "Politique", "economie": "Économie", "education": "Éducation",
        "sante": "Santé", "sport": "Sport", "societe": "Société"}

# Résultats du notebook 02_modeles.ipynb (F1 par thème, validation croisée 5 plis)
F1_PAR_THEME = pd.DataFrame({
    "theme": ["economie", "education", "politique", "sante", "societe", "sport"],
    "Baseline (mots)": [0.38, 0.67, 0.62, 0.76, 0.57, 0.56],
    "Approche 2 (caractères)": [0.52, 0.75, 0.70, 0.75, 0.65, 0.49],
})

EXEMPLES = [
    "Le ministère de la Santé lance une campagne contre la dengue à Balbala",
    "Athlétisme : un Djiboutien remporte le marathon de Nairobi",
    "Hausse des exportations via le corridor Djibouti-Éthiopie",
    "Rentrée scolaire : 5 000 nouveaux élèves accueillis à Ali-Sabieh",
    "Le Président Guelleh reçoit l'ambassadeur de France",
    "Festival culturel afar à Tadjourah : musique et danses traditionnelles",
    "Visite officielle du Premier ministre éthiopien à Djibouti",
]

# Icônes au trait (SVG), sans émojis
ICONES = {
    "doc": '<path d="M7 3h7l5 5v13H7z"/><path d="M14 3v5h5"/><path d="M9.5 13h7M9.5 17h7"/>',
    "cible": '<circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="4"/><circle cx="12" cy="12" r="0.8"/>',
    "fleche": '<path d="M4 17l6-6 4 4 6-7"/><path d="M15 8h5v5"/>',
    "grille": '<rect x="4" y="4" width="7" height="7" rx="1"/><rect x="13" y="4" width="7" height="7" rx="1"/><rect x="4" y="13" width="7" height="7" rx="1"/><rect x="13" y="13" width="7" height="7" rx="1"/>',
    "crayon": '<path d="M4 20h4L19 9l-4-4L4 16z"/><path d="M13 7l4 4"/>',
    "plis": '<path d="M4 6h16M4 12h16M4 18h16"/><path d="M9 4v16"/>',
    "loupe": '<circle cx="11" cy="11" r="6"/><path d="M20 20l-4.5-4.5"/>',
    "calendrier": '<rect x="4" y="5" width="16" height="15" rx="2"/><path d="M4 10h16M9 3v4M15 3v4"/>',
}


def icone(nom, taille=26):
    return (f'<svg width="{taille}" height="{taille}" viewBox="0 0 24 24" fill="none" stroke="white" '
            f'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">{ICONES[nom]}</svg>')


# ---------------------------------------------------------------- 2. Style
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, .stApp, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
.stApp {{ background: #F4F7FB; }}
.block-container {{ padding-top: 1.2rem; max-width: 1250px; }}
header[data-testid="stHeader"] {{ background: transparent; }}

.bandeau {{ display: flex; align-items: center; gap: 1.2rem; background: linear-gradient(180deg, {MARINE}, {BLEU}); color: white;
           border-radius: 0 0 22px 22px; padding: .9rem 1.4rem; margin: 0 0 1.4rem;
           box-shadow: 0 6px 18px rgba(11,46,89,.18); }}
.bandeau .logo {{ background: white; border-radius: 12px; padding: .35rem .5rem; flex-shrink: 0; }}
.bandeau .logo img {{ height: 76px; display: block; }}
.texte-bandeau {{ flex: 1; text-align: center; }}
.bandeau h1 {{ color: white; font-size: 1.7rem; font-weight: 800; letter-spacing: .02em; margin: 0; padding: 0; }}
.bandeau p {{ color: {BLEU_CLAIR}; margin: .25rem 0 0; font-size: .92rem; }}

div[class*="st-key-carte"] {{ background: white; border: 1px solid #DCE6F2 !important; border-radius: 12px;
                              box-shadow: 0 2px 8px rgba(11,46,89,.06); padding: .4rem .6rem; }}
.titre-carte {{ color: {MARINE}; font-weight: 700; font-size: .82rem; letter-spacing: .06em;
               text-transform: uppercase; text-align: center; margin: .2rem 0 .6rem; }}

.kpi {{ display: flex; align-items: center; gap: .9rem; background: white; border: 1px solid #DCE6F2;
        border-radius: 12px; padding: .9rem 1rem; box-shadow: 0 2px 8px rgba(11,46,89,.06); min-height: 118px; }}
.rond {{ background: {MARINE}; border-radius: 50%; width: 52px; height: 52px; min-width: 52px;
         display: flex; align-items: center; justify-content: center; }}
.kpi .label {{ color: {MARINE}; font-weight: 700; font-size: .74rem; letter-spacing: .06em; text-transform: uppercase; }}
.kpi .valeur {{ color: {MARINE}; font-weight: 800; font-size: 1.75rem; line-height: 1.15; }}
.kpi .detail {{ color: {GRIS_TEXTE}; font-size: .8rem; }}

.resultat {{ background: {BLEU_PALE}; border-radius: 10px; padding: .8rem 1rem; margin: .6rem 0; }}
.resultat .label {{ color: {GRIS_TEXTE}; font-size: .74rem; letter-spacing: .06em; text-transform: uppercase; }}
.resultat .theme {{ color: {MARINE}; font-size: 1.8rem; font-weight: 800; }}
.resultat .proba {{ color: {GRIS_TEXTE}; font-size: .88rem; }}

.stButton button {{ background: {MARINE}; color: white; border: none; border-radius: 8px; font-weight: 600; }}
.stButton button:hover {{ background: {BLEU}; color: white; }}
.stTextArea textarea {{ background: #FAFCFE; border: 1px solid #C9D8EA; }}

.pied {{ display: flex; flex-wrap: wrap; justify-content: space-between; gap: 1rem; background: white;
         border: 1px solid #DCE6F2; border-radius: 12px; padding: .9rem 1.2rem; margin-top: 1.2rem; }}
.pied .bloc {{ display: flex; align-items: center; gap: .7rem; min-width: 200px; flex: 1; }}
.pied .rond {{ width: 44px; height: 44px; min-width: 44px; }}
.pied .label {{ color: {MARINE}; font-weight: 700; font-size: .72rem; letter-spacing: .05em; text-transform: uppercase; }}
.pied .valeur {{ color: {MARINE}; font-weight: 800; font-size: 1.2rem; }}
.pied .detail {{ color: {GRIS_TEXTE}; font-size: .76rem; }}
.note {{ text-align: center; color: {GRIS_TEXTE}; font-size: .8rem; margin-top: .8rem; }}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- 3. Données et modèle
@st.cache_resource
def charger_modele():
    return joblib.load("modele.joblib")


@st.cache_data
def charger_corpus():
    corpus = pd.read_csv("data/corpus.csv")
    return corpus[corpus["remarque"].fillna("") != "exclure"]


modele = charger_modele()
corpus = charger_corpus()
comparaison = pd.read_csv("resultats/comparaison.csv")
f1_base = comparaison.iloc[0]["Macro-F1 moyenne"]
f1_car = comparaison.iloc[1]["Macro-F1 moyenne"]


def kpi(nom_icone, label, valeur, detail):
    return (f'<div class="kpi"><div class="rond">{icone(nom_icone)}</div><div>'
            f'<div class="label">{label}</div><div class="valeur">{valeur}</div>'
            f'<div class="detail">{detail}</div></div></div>')


def virgule(x, decimales=3):
    return f"{x:.{decimales}f}".replace(".", ",")


# ---------------------------------------------------------------- 4. Bandeau
with open("assets/logo_ud.jpg", "rb") as f:
    logo = base64.b64encode(f.read()).decode()   # le logo est inséré dans la page sous forme de texte

st.markdown(f"""
<div class="bandeau">
  <div class="logo"><img src="data:image/jpeg;base64,{logo}" alt="Université de Djibouti"></div>
  <div class="texte-bandeau">
    <h1>CLASSIFICATION THÉMATIQUE D'ARTICLES DJIBOUTIENS</h1>
    <p>Projet TALN · Master 2 IAMD · Université de Djibouti</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------- 5. Chiffres clés
c1, c2, c3, c4 = st.columns(4, gap="medium")
c1.markdown(kpi("doc", "Titres annotés", len(corpus), "La Nation et ADI, vérifiés à la main"), unsafe_allow_html=True)
c2.markdown(kpi("cible", "Macro-F1", virgule(f1_car), "Approche 2, validation croisée"), unsafe_allow_html=True)
c3.markdown(kpi("fleche", "Gain sur la baseline", "+" + virgule(f1_car - f1_base), f"Baseline : {virgule(f1_base)}"),
            unsafe_allow_html=True)
c4.markdown(kpi("grille", "Thèmes", corpus["theme_final"].nunique(), "Politique à société"), unsafe_allow_html=True)
st.write("")

# ---------------------------------------------------------------- 6. Test en direct + répartition du corpus
gauche, droite = st.columns([3, 2], gap="medium")

with gauche.container(key="carte_test"):
    st.markdown('<div class="titre-carte">Tester un titre</div>', unsafe_allow_html=True)
    exemple = st.selectbox("Choisir un exemple", ["Saisir mon propre titre"] + EXEMPLES)
    texte = st.text_area("Titre ou court article",
                         value="" if exemple == "Saisir mon propre titre" else exemple, height=90)

    if st.button("Trouver le thème", use_container_width=True):
        st.session_state["texte_teste"] = texte

    texte_teste = st.session_state.get("texte_teste", "")
    if texte_teste.strip():
        probas = pd.Series(modele.predict_proba([texte_teste])[0], index=modele.classes_).sort_values(ascending=False)
        st.markdown(f"""
        <div class="resultat">
          <div class="label">Thème prédit</div>
          <div class="theme">{NOMS[probas.index[0]]}</div>
          <div class="proba">Probabilité : {probas.iloc[0] * 100:.0f} %, contre {probas.iloc[1] * 100:.0f} % pour
          {NOMS[probas.index[1]]}</div>
        </div>""", unsafe_allow_html=True)

        donnees = pd.DataFrame({"Thème": [NOMS[t] for t in probas.index], "Probabilité": probas.values,
                                "Premier": [i == 0 for i in range(len(probas))]})
        barres = alt.Chart(donnees).mark_bar(cornerRadiusEnd=4, height=16).encode(
            x=alt.X("Probabilité:Q", axis=alt.Axis(format="%", grid=True, title=None), scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("Thème:N", sort=None, title=None),
            color=alt.condition("datum.Premier", alt.value(MARINE), alt.value(BLEU_CLAIR)),
            tooltip=[alt.Tooltip("Probabilité:Q", format=".0%")],
        ).properties(height=210)
        st.altair_chart(barres, use_container_width=True)

        if probas.iloc[0] - probas.iloc[1] < 0.10:
            st.info(f"Titre ambigu : {NOMS[probas.index[0]]} et {NOMS[probas.index[1]]} sont presque à égalité.")
    else:
        st.caption("Choisissez un exemple ou collez un titre, puis cliquez sur « Trouver le thème ».")

with droite.container(key="carte_corpus"):
    st.markdown('<div class="titre-carte">Répartition du corpus par thème</div>', unsafe_allow_html=True)
    repartition = corpus["theme_final"].map(NOMS).value_counts().rename_axis("Thème").reset_index(name="Titres")
    repartition["Part"] = repartition["Titres"] / repartition["Titres"].sum()
    anneau = alt.Chart(repartition).mark_arc(innerRadius=62, outerRadius=118, stroke="white", strokeWidth=2).encode(
        theta=alt.Theta("Titres:Q"),
        color=alt.Color("Thème:N", sort=list(repartition["Thème"]), scale=alt.Scale(range=PALETTE),
                        legend=alt.Legend(title=None, orient="right", labelFontSize=12)),
        tooltip=["Thème", "Titres", alt.Tooltip("Part:Q", format=".1%")],
    ).properties(height=290)
    st.altair_chart(anneau, use_container_width=True)
    st.caption("Entre 36 et 43 titres par thème : un corpus équilibré.")

st.write("")

# ---------------------------------------------------------------- 7. Comparaison des approches + sources
bas_gauche, bas_droite = st.columns([3, 2], gap="medium")

with bas_gauche.container(key="carte_f1"):
    st.markdown('<div class="titre-carte">F1 par thème : baseline et approche 2</div>', unsafe_allow_html=True)
    f1 = F1_PAR_THEME.assign(Thème=F1_PAR_THEME["theme"].map(NOMS)).melt(
        id_vars="Thème", value_vars=["Baseline (mots)", "Approche 2 (caractères)"], var_name="Approche", value_name="F1")
    groupes = alt.Chart(f1).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
        x=alt.X("Thème:N", title=None, axis=alt.Axis(labelAngle=0)),
        xOffset=alt.XOffset("Approche:N", sort=["Baseline (mots)", "Approche 2 (caractères)"]),
        y=alt.Y("F1:Q", scale=alt.Scale(domain=[0, 1]), title="F1"),
        color=alt.Color("Approche:N", sort=["Baseline (mots)", "Approche 2 (caractères)"],
                        scale=alt.Scale(range=[BLEU_CLAIR, MARINE]), legend=alt.Legend(title=None, orient="top")),
        tooltip=["Thème", "Approche", alt.Tooltip("F1:Q", format=".2f")],
    ).properties(height=280)
    st.altair_chart(groupes, use_container_width=True)

with bas_droite.container(key="carte_sources"):
    st.markdown('<div class="titre-carte">Titres par source et par thème</div>', unsafe_allow_html=True)
    tableau = pd.crosstab(corpus["theme_final"].map(NOMS), corpus["source"]).rename_axis(None).rename_axis(None, axis=1)
    tableau["Total"] = tableau.sum(axis=1)
    st.dataframe(tableau, use_container_width=True)
    st.caption("Éducation vient presque entièrement de La Nation (40 sur 43), Société presque entièrement "
               "de l'ADI (39 sur 42) : un biais de source que l'analyse des erreurs a mis en évidence.")

# ---------------------------------------------------------------- 8. Bande du bas
corriges = (corpus["theme_final"] != corpus["theme_rubrique"]).mean() * 100
st.markdown(f"""
<div class="pied">
  <div class="bloc"><div class="rond">{icone("crayon", 22)}</div><div><div class="label">Thèmes corrigés</div>
       <div class="valeur">{virgule(corriges, 1)} %</div><div class="detail">Rubrique du site corrigée à la main</div></div></div>
  <div class="bloc"><div class="rond">{icone("plis", 22)}</div><div><div class="label">Plis gagnés</div>
       <div class="valeur">4 sur 5</div><div class="detail">Approche 2 contre baseline</div></div></div>
  <div class="bloc"><div class="rond">{icone("loupe", 22)}</div><div><div class="label">Erreurs analysées</div>
       <div class="valeur">14</div><div class="detail">Regroupées en 6 causes</div></div></div>
  <div class="bloc"><div class="rond">{icone("calendrier", 22)}</div><div><div class="label">Corpus collecté</div>
       <div class="valeur">19 nov. 2026</div><div class="detail">lanation.dj et adi.dj</div></div></div>
</div>
<p class="note">Modèle : TF-IDF sur n-grammes de caractères (2 à 5) et régression logistique, entraîné sur les titres
du corpus. Conçu pour des titres : un long article est accepté, mais le modèle n'a appris que sur des titres.</p>
""", unsafe_allow_html=True)
