# app.py — Tableau de bord du complexe portuaire de San Pedro Bay (Los Angeles + Long Beach)
# Lancer :  streamlit run app.py
#
# Organisation du fichier :
#   1. Réglages (seuils, coordonnées)
#   2. Chargement des données (fichiers CSV + API météo + AIS), avec message si une source manque
#   3. Calculs : série totale, prévision avec marge d'incertitude, anomalies
#   4. Style (violet et rose)        5. En-tête
#   6. Les 7 tuiles (valeur, référence, tendance, fraîcheur)
#   7. Volumes   8. Comparaison mondiale   9. En direct   10. Recommandations
#   11. Méthode, limites et bande de traçabilité des sources

import base64
import os
from datetime import datetime, timezone

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# ---------------------------------------------------------------- 1. Réglages
st.set_page_config(page_title="San Pedro Bay — Tableau de bord", page_icon=":material/directions_boat:", layout="wide")

LAT, LON = 33.73, -118.26          # entrée du port, entre Los Angeles et Long Beach
RAFALE_VIGILANCE = 40              # km/h — seuils choisis par nous, à valider avec la capitainerie
RAFALE_ALERTE = 60                 # km/h
VAGUE_ALERTE = 2.0                 # mètres
SEUIL_ANOMALIE = 2                 # un mois est anormal si son écart dépasse 2 écarts-types


# ---------------------------------------------------------------- 2. Chargement
@st.cache_data
def lire_csv(nom):
    # Lit un fichier du dossier data/. Renvoie un tableau vide si le fichier manque.
    try:
        return pd.read_csv(f"data/{nom}")
    except FileNotFoundError:
        return pd.DataFrame()


@st.cache_data(ttl=1800)            # l'API n'est rappelée qu'une fois toutes les 30 minutes
def lire_meteo():
    vent = requests.get("https://api.open-meteo.com/v1/forecast", timeout=15, params={
        "latitude": LAT, "longitude": LON, "timezone": "America/Los_Angeles", "forecast_days": 3,
        "current": "wind_speed_10m,wind_gusts_10m",
        "hourly": "wind_speed_10m,wind_gusts_10m"}).json()
    mer = requests.get("https://marine-api.open-meteo.com/v1/marine", timeout=15, params={
        "latitude": 33.70, "longitude": -118.25, "timezone": "America/Los_Angeles", "forecast_days": 3,
        "current": "wave_height", "hourly": "wave_height"}).json()
    horaire = pd.DataFrame({
        "heure": pd.to_datetime(vent["hourly"]["time"]),
        "vent": vent["hourly"]["wind_speed_10m"],
        "rafales": vent["hourly"]["wind_gusts_10m"],
        "vagues": mer["hourly"]["wave_height"],
    })
    actuel = {"vent": vent["current"]["wind_speed_10m"], "rafales": vent["current"]["wind_gusts_10m"],
              "vagues": mer["current"]["wave_height"], "heure": vent["current"]["time"]}
    return actuel, horaire


@st.cache_data(ttl=600)             # au plus une capture AIS toutes les 10 minutes
def lire_ais_en_direct(cle):
    from ais_snapshot import capturer
    return capturer(cle, duree=30)


def lire_ais():
    # En ligne : la clé est dans les Secrets Streamlit -> capture en direct.
    # Sinon : on lit la dernière capture enregistrée par ais_snapshot.py.
    try:
        cle = st.secrets["AIS_API_KEY"]
    except Exception:
        cle = None
    if cle:
        try:
            return lire_ais_en_direct(cle), "direct"
        except Exception:
            pass
    return lire_csv("ais_snapshot.csv"), "fichier"


sources = lire_csv("sources.csv")
pola = lire_csv("volumes_pola.csv")
polb = lire_csv("volumes_polb.csv")
cppi = lire_csv("cppi.csv")
cnuced = lire_csv("cnuced.csv")


def date_source(mot):
    # Date de récupération inscrite dans data/sources.csv pour la source qui contient ce mot
    if sources.empty:
        return "?"
    ligne = sources[sources["source"].str.contains(mot)]
    return ligne["date_recuperation"].iloc[0] if len(ligne) else "?"


# ---------------------------------------------------------------- 3. Calculs
for df in (pola, polb):
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])

# Série totale du complexe = Los Angeles + Long Beach, mois par mois (mois communs aux deux ports)
if not pola.empty and not polb.empty:
    total = pd.merge(pola[["date", "total"]], polb[["date", "total"]], on="date", suffixes=("_la", "_lb"))
    total["total"] = total["total_la"] + total["total_lb"]
    perimetre = "Los Angeles + Long Beach"
elif not pola.empty:
    total = pola[["date", "total"]].copy()
    perimetre = "Los Angeles seul (Long Beach indisponible)"
else:
    total = pd.DataFrame()
    perimetre = "aucune donnée"

serie = total.set_index("date")["total"].asfreq("MS") if not total.empty else pd.Series(dtype=float)
# Un mois absent de la source crée un trou : on le comble par interpolation et on le signale à l'écran
mois_manquants = list(serie[serie.isna()].index)
serie = serie.interpolate()

prevision = None
if len(serie) >= 48:
    # a) On mesure l'erreur du modèle sur les 12 derniers mois connus (comme en TP : MAE)
    appr, valid = serie[:-12], serie[-12:]
    modele_test = ExponentialSmoothing(appr, trend="add", seasonal="add", seasonal_periods=12).fit()
    mae = (modele_test.forecast(12) - valid).abs().mean()
    # b) Référence à battre : « même mois de l'année précédente »
    mae_reference = (serie.shift(12)[-12:] - valid).abs().mean()
    # c) On réentraîne sur tout l'historique et on prévoit 6 mois, avec une fourchette ± MAE
    modele = ExponentialSmoothing(serie, trend="add", seasonal="add", seasonal_periods=12).fit()
    futur = modele.forecast(6)
    prevision = pd.DataFrame({"date": futur.index, "prevu": futur.values,
                              "bas": futur.values - mae, "haut": futur.values + mae})
    # d) Anomalies : mois où l'écart entre le réel et le modèle dépasse 2 écarts-types
    ecart = serie - modele.fittedvalues
    score = (ecart - ecart.mean()) / ecart.std()
    anomalies = serie[score.abs() > SEUIL_ANOMALIE]

try:
    meteo, meteo_h = lire_meteo()
except Exception:
    meteo, meteo_h = None, None

ais, mode_ais = lire_ais()

# ---------------------------------------------------------------- 4. Style (couleurs violet et rose)
VIOLET_FONCE, VIOLET, MAUVE, ROSE, ROSE_CLAIR = "#2E1746", "#5B2A86", "#B57EDC", "#D9468A", "#F6D5E5"
VERT, GRIS = "#23845A", "#6E6280"
COULEURS = [VIOLET, ROSE, MAUVE, "#F0A3C4", VIOLET_FONCE, "#9D8AB0"]
px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = COULEURS

# Photo de fond : facultative (assets/bandeau.jpg). Elle couvre toute la page, floutée sous un voile violet.
CREDIT_BANDEAU = "assets/credit.txt"
IMAGE_FOND = next((f"assets/bandeau.{ext}" for ext in ("jpg", "jpeg", "png")
                   if os.path.exists(f"assets/bandeau.{ext}")), None)
if IMAGE_FOND:
    with open(IMAGE_FOND, "rb") as f:
        photo = base64.b64encode(f.read()).decode()
    FOND_PAGE = f"url(data:image/jpeg;base64,{photo}) center / cover"
else:
    FOND_PAGE = f"linear-gradient(135deg, {VIOLET_FONCE}, {VIOLET} 60%, {ROSE})"

# Logo : facultatif (assets/logo.jpg), affiché à gauche du titre
LOGO = ""
if os.path.exists("assets/logo.jpg"):
    with open("assets/logo.jpg", "rb") as f:
        LOGO = f'<img class="logo" src="data:image/jpeg;base64,{base64.b64encode(f.read()).decode()}">'

st.markdown(f"""
<style>
.stApp {{ background: {VIOLET_FONCE}; }}
.stApp::before {{ content: ""; position: fixed; inset: 0; z-index: 0; background: {FOND_PAGE};
                  filter: saturate(1.15) contrast(1.05); }}
.stApp::after {{ content: ""; position: fixed; inset: 0; z-index: 0;
                 background: linear-gradient(160deg, rgba(46,23,70,0.78) 0%, rgba(74,32,110,0.62) 45%,
                                             rgba(160,50,120,0.50) 100%); }}
[data-testid="stAppViewContainer"], [data-testid="stMain"] {{ position: relative; z-index: 1; background: transparent; }}
.block-container {{ padding-top: 1.2rem; max-width: 1400px; }}
header[data-testid="stHeader"] {{ display: none; }}
.bandeau {{ border-radius: 18px; margin-bottom: 24px; overflow: hidden;
            background: linear-gradient(110deg, rgba(46,23,70,0.60), rgba(91,42,134,0.40) 60%, rgba(217,70,138,0.40));
            border: 1px solid rgba(255,255,255,0.22); backdrop-filter: blur(6px);
            box-shadow: 0 12px 32px rgba(0,0,0,0.30); }}
.bandeau .ruban {{ display: flex; justify-content: space-between; gap: 16px; flex-wrap: wrap;
                   padding: 9px 26px; background: rgba(20,8,34,0.45); border-bottom: 1px solid rgba(255,255,255,0.15);
                   color: #EFE3F7; font-size: 11px; font-weight: 600; letter-spacing: 2px; text-transform: uppercase; }}
.bandeau .ruban b {{ color: white; }}
.bandeau .corps {{ display: grid; grid-template-columns: auto minmax(0, 1fr) minmax(260px, 340px);
                   gap: 28px; align-items: center; padding: 24px 26px; }}
.bandeau .logo {{ width: 200px; height: 125px; object-fit: cover; border-radius: 12px; display: block;
                  border: 2px solid rgba(255,255,255,0.75); box-shadow: 0 6px 18px rgba(0,0,0,0.40); }}
.bandeau .surtitre {{ color: {ROSE_CLAIR}; font-size: 12px; font-weight: 700; letter-spacing: 3px; text-transform: uppercase; }}
.bandeau h1 {{ color: white; font-size: 36px; font-weight: 800; margin: 6px 0 8px 0; padding: 0; line-height: 1.1;
               text-shadow: 0 2px 10px rgba(0,0,0,0.35); }}
.bandeau .accroche {{ color: #F3E9F8; font-size: 14px; max-width: 560px; }}
.bandeau .fiche {{ border-left: 3px solid {ROSE}; padding-left: 16px; display: grid; gap: 10px; }}
.bandeau .fiche .etiquette {{ color: {ROSE_CLAIR}; font-size: 10px; font-weight: 700; letter-spacing: 2px; }}
.bandeau .fiche .texte {{ color: white; font-size: 14px; font-weight: 600; }}
.tuile {{ background: white; border-radius: 12px; padding: 16px 18px; min-height: 225px;
          box-shadow: 0 8px 24px rgba(0,0,0,0.25); border-left: 4px solid {ROSE}; }}
.tuile .haut {{ display: flex; align-items: center; gap: 10px; }}
.tuile .icone {{ background: {VIOLET}; border-radius: 10px; width: 36px; height: 36px; flex-shrink: 0;
                 display: flex; align-items: center; justify-content: center; }}
.tuile .titre {{ font-size: 11px; font-weight: 700; color: {GRIS}; text-transform: uppercase; letter-spacing: 0.6px; }}
.tuile .valeur {{ font-size: 30px; font-weight: 800; color: {VIOLET_FONCE}; margin: 8px 0 4px 0; }}
.tuile .pastille {{ display: inline-block; border-radius: 4px; padding: 1px 8px; font-size: 12px; font-weight: 700; }}
.tuile .ref {{ font-size: 12px; color: #3d3d3d; margin-top: 8px; }}
.tuile .source {{ font-size: 11px; color: {GRIS}; margin-top: 2px; }}
.section {{ color: white; font-size: 15px; font-weight: 800; margin: 22px 0 8px 0; text-transform: uppercase;
            letter-spacing: 1px; border-left: 4px solid {ROSE}; padding-left: 10px; }}
div[class*="st-key-carte_"] {{ background: white; border-radius: 12px; padding: 6px 10px;
                               box-shadow: 0 8px 24px rgba(0,0,0,0.25); }}
div[data-testid="stExpander"] details {{ background: white; border-radius: 12px; }}
.note {{ color: #EFE3F7; font-size: 13px; margin: -2px 0 10px 0; }}
.note b {{ color: white; }}
.lecture {{ background: #F7F0FA; border-left: 3px solid {ROSE}; border-radius: 6px; padding: 8px 12px;
            font-size: 12.5px; color: #2b2b2b; margin: 4px 0 6px 0; }}
.lecture b {{ color: {VIOLET}; }}
.trace {{ background: white; border-radius: 12px; padding: 12px 18px; margin-top: 18px; border-left: 4px solid {ROSE};
          box-shadow: 0 1px 6px rgba(46,23,70,0.08); font-size: 12px; color: #333; }}
.trace b {{ color: {VIOLET}; }}
</style>""", unsafe_allow_html=True)

# Icônes au trait (dessinées en SVG, sans emoji)
TRAITS = {
    "volume": '<path d="M3 7l9-4 9 4v10l-9 4-9-4z"/><path d="M3 7l9 4 9-4M12 11v10"/>',
    "vides": '<path d="M4 12a8 8 0 0 1 14-5l2 2"/><path d="M20 4v5h-5"/><path d="M20 12a8 8 0 0 1-14 5l-2-2"/><path d="M4 20v-5h5"/>',
    "prevision": '<path d="M3 17l6-6 4 4 8-8"/><path d="M15 7h6v6"/>',
    "rang": '<path d="M5 20V11M12 20V5M19 20v-6M2 20h20"/>',
    "temps": '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
    "navire": '<path d="M3 15h18l-2.5 5h-13z"/><path d="M6 15V10h12v5M10 10V6h4v4"/>',
    "vent": '<path d="M3 8h11a3 3 0 1 0-3-3"/><path d="M3 12h16a3 3 0 1 1-3 3"/><path d="M3 16h7"/>',
}


def icone(nom):
    return (f'<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" '
            f'stroke-linecap="round" stroke-linejoin="round">{TRAITS[nom]}</svg>')


def tuile(nom_icone, titre, valeur, tendance, sens, reference, source):
    # Une tuile = les 4 éléments du cours : valeur, référence, tendance, fraîcheur (source + date).
    # sens : "bon" (vert), "mauvais" (rose) ou "neutre" (gris) -> couleur de la tendance
    couleur = {"bon": VERT, "mauvais": ROSE, "neutre": GRIS}[sens]
    pastille = (f'<span class="pastille" style="background:{couleur}1F;color:{couleur}">{tendance}</span>'
                if tendance else "")
    st.markdown(f"""<div class="tuile"><div class="haut"><div class="icone">{icone(nom_icone)}</div>
        <div class="titre">{titre}</div></div><div class="valeur">{valeur}</div>{pastille}
        <div class="ref">Réf. : {reference}</div><div class="source">{source}</div></div>""",
                unsafe_allow_html=True)


def tuile_vide(nom_icone, titre, message):
    tuile(nom_icone, titre, "—", None, "neutre", message, "source indisponible")


def section(texte):
    st.markdown(f'<div class="section">{texte}</div>', unsafe_allow_html=True)


def lecture(texte):
    # Interprétation calculée à partir des données : elle se met à jour avec elles
    st.markdown(f'<div class="lecture"><b>Lecture.</b> {texte}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------- 5. En-tête
st.markdown(f"""<div class="bandeau">
  <div class="ruban"><span>Université de Djibouti · M2 IAMD · IA sectorielle</span>
    <span><b>Fatouma Soubere Mahamoud</b> · mis à jour le {datetime.now(timezone.utc):%d/%m/%Y}</span></div>
  <div class="corps">
    <div>{LOGO}</div>
    <div><div class="surtitre">Complexe portuaire · Los Angeles &amp; Long Beach</div>
      <h1>San Pedro Bay</h1>
      <div class="accroche">Tableau de bord portuaire : activité du complexe, prévision à 6 mois,
        situation en direct et position face aux ports du monde.</div></div>
    <div class="fiche">
      <div><div class="etiquette">UTILISATEUR</div><div class="texte">Planification de l'autorité portuaire</div></div>
      <div><div class="etiquette">DÉCISION</div><div class="texte">Moyens du mois à venir</div></div>
      <div><div class="etiquette">COMPARAISON</div><div class="texte">CPPI : 7 ports, dont Djibouti</div></div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

# ---------------------------------------------------------------- 6. Les 7 tuiles
c1, c2, c3, c4 = st.columns(4, gap="medium")
st.markdown('<div style="height:18px"></div>', unsafe_allow_html=True)
c5, c6, c7 = st.columns(3, gap="medium")

with c1:   # 1 — volume du dernier mois vs même mois N-1
    if len(serie) >= 13:
        dernier, mois = serie.iloc[-1], serie.index[-1]
        ref = serie.get(mois - pd.DateOffset(years=1))
        evol = (dernier / ref - 1) * 100
        tuile("volume", f"EVP traités · {mois:%m/%Y}", f"{dernier/1e6:.2f} M",
              f"{'▲' if evol >= 0 else '▼'} {evol:+.1f} % sur 1 an", "bon" if evol >= 0 else "mauvais",
              f"même mois N-1 = {ref/1e6:.2f} M EVP",
              f"Port of Los Angeles · récupéré le {date_source('Los Angeles')}")
    else:
        tuile_vide("volume", "EVP traités", "lancer recuperer_volumes.py")

with c2:   # 2 — part des vides vs moyenne 2019
    if not pola.empty:
        vides = pd.concat([pola, polb]) if not polb.empty else pola
        par_mois = vides.groupby("date")[["vides", "total"]].sum()
        part = par_mois["vides"] / par_mois["total"] * 100
        moyenne = part[part.index.year < 2020].mean()
        ecart_vides = part.iloc[-1] - moyenne
        tuile("vides", "Part des EVP vides", f"{part.iloc[-1]:.1f} %",
              f"{'▲' if ecart_vides >= 0 else '▼'} {ecart_vides:+.1f} pt vs 2019",
              "mauvais" if ecart_vides > 2 else "neutre", f"moyenne 2019 = {moyenne:.1f} % (avant Covid)",
              f"Port of Los Angeles · récupéré le {date_source('Los Angeles')}")
    else:
        tuile_vide("vides", "Part des EVP vides", "volumes manquants")

with c3:   # 3 — prévision du mois suivant ± MAE
    if prevision is not None:
        p = prevision.iloc[0]
        evol = (p["prevu"] / serie.iloc[-1] - 1) * 100
        tuile("prevision", f"Prévision · {p['date']:%m/%Y}", f"{p['prevu']/1e6:.2f} M",
              f"entre {p['bas']/1e6:.2f} et {p['haut']/1e6:.2f} M", "neutre",
              f"erreur moyenne du modèle = {mae/1e3:.0f} k EVP (MAE, 12 mois)",
              "Holt-Winters · recalculé à chaque mise à jour des volumes")
    else:
        tuile_vide("prevision", "Prévision", "moins de 4 ans de données")

with c4:   # 4 — rang CPPI de Los Angeles vs Djibouti
    la = cppi[cppi["port"] == "Los Angeles"].sort_values("edition") if not cppi.empty else cppi
    if len(la) >= 2:
        avant, apres = la.iloc[-2], la.iloc[-1]
        gain = int(avant["rang"] - apres["rang"])
        djib = cppi[(cppi["port"] == "Djibouti") & (cppi["edition"] == apres["edition"])]
        tuile("rang", f"Rang CPPI {apres['edition']} · Los Angeles", f"{int(apres['rang'])}e",
              f"{'▲' if gain >= 0 else '▼'} {gain:+d} places vs {avant['edition']}", "bon" if gain >= 0 else "mauvais",
              f"Djibouti {int(djib['rang'].iloc[0])}e · 1 = meilleur port" if len(djib) else "1 = meilleur port",
              f"Banque mondiale, CPPI · récupéré le {date_source('CPPI 2024')}")
    else:
        tuile_vide("rang", "Rang CPPI", "remplir data/cppi.csv")

with c5:   # 5 — temps médian au port, États-Unis vs monde
    if not cnuced.empty:
        an = cnuced["annee"].max()
        dernier_an = cnuced[cnuced["annee"] == an].set_index("pays")
        avant_an = cnuced[cnuced["annee"] == an - 1].set_index("pays")
        usa = dernier_an.loc["États-Unis", "temps_median_jours"]
        monde = dernier_an.loc["Monde", "temps_median_jours"]
        evol = usa - avant_an.loc["États-Unis", "temps_median_jours"]
        tuile("temps", f"Temps médian au port · États-Unis {an}", f"{usa:.2f} jour",
              f"{'▼' if evol <= 0 else '▲'} {evol:+.2f} jour vs {an - 1}", "bon" if evol <= 0 else "mauvais",
              f"moyenne mondiale = {monde:.2f} jour (porte-conteneurs)",
              f"CNUCED, UNCTADstat · récupéré le {date_source('CNUCED')}")
    else:
        tuile_vide("temps", "Temps médian au port", "remplir data/cnuced.csv")

with c6:   # 6 — navires dans la zone (AIS)
    if not ais.empty and "lat" in ais.columns:
        dans_zone = ais[ais["lat"].between(33.55, 33.80) & ais["lon"].between(-118.45, -118.05)]
        mouillage = int((dans_zone["statut"] == "Au mouillage").sum())
        tuile("navire", "Navires dans la zone · AIS", f"{len(dans_zone)}", f"dont {mouillage} au mouillage", "neutre",
              "un mouillage qui se remplit signale une attente avant quai",
              f"aisstream.io · capture {ais['capture_utc'].iloc[0]} ({mode_ais})")
    else:
        tuile_vide("navire", "Navires dans la zone · AIS", "aucune capture récente")

with c7:   # 7 — rafales max prévues sur 48 h (alerte météo)
    if meteo is not None:
        maxi = meteo_h["rafales"][:48].max()
        if maxi >= RAFALE_ALERTE:
            etat, sens = "ALERTE", "mauvais"
        elif maxi >= RAFALE_VIGILANCE:
            etat, sens = "VIGILANCE", "mauvais"
        else:
            etat, sens = "SOUS LES SEUILS", "bon"
        tuile("vent", "Rafales max. prévues · 48 h", f"{maxi:.0f} km/h", etat, sens,
              f"vigilance {RAFALE_VIGILANCE} km/h · alerte {RAFALE_ALERTE} km/h",
              f"Open-Meteo · mis à jour {meteo['heure']} (heure locale)")
    else:
        tuile_vide("vent", "Rafales max. prévues · 48 h", "API météo injoignable")


def joli(fig, hauteur=330):
    # Mise en forme commune à tous les graphiques
    fig.update_layout(height=hauteur, margin=dict(l=10, r=10, t=45, b=10), font=dict(color=VIOLET_FONCE),
                      title_font=dict(size=14, color=VIOLET_FONCE), legend_title_text="")
    return fig


# ---------------------------------------------------------------- 7. Volumes
section("Volumes de Los Angeles et Long Beach")
if pola.empty:
    st.warning("Volumes indisponibles : lancer recuperer_volumes.py.")
else:
    if polb.empty:
        st.info("Long Beach indisponible : le site officiel (polb.com) refuse l'accès depuis Djibouti "
                "(« Access Denied », 18/11/2026) et sa copie archivée ne contient pas les tableaux. "
                "Les volumes affichés concernent Los Angeles seul.")
    deux = pd.concat([pola.assign(port="Los Angeles"), polb.assign(port="Long Beach")])
    g, d = st.columns(2)
    with g.container(border=True, key="carte_1"):
        st.plotly_chart(joli(px.line(deux, x="date", y="total", color="port", labels={"total": "EVP", "date": ""},
                                     title="EVP PAR MOIS DEPUIS 2019")), width="stretch")
        douze, avant = serie[-12:].sum(), serie[-24:-12].sum()
        pic = serie.idxmax()
        lecture(f"Sur les 12 derniers mois, Los Angeles a traité {douze/1e6:.2f} M EVP, soit "
                f"{(douze/avant - 1)*100:+.1f} % par rapport aux 12 mois précédents. Le record de la période est "
                f"{pic:%m/%Y} ({serie.max()/1e6:.2f} M EVP). L'activité suit un cycle annuel marqué, avec des creux "
                "en début d'année, ce qui justifie un modèle saisonnier.")
    with d.container(border=True, key="carte_2"):
        detail = pola.melt(id_vars="date", value_vars=["imports", "exports", "vides"], var_name="flux", value_name="EVP")
        st.plotly_chart(joli(px.bar(detail, x="date", y="EVP", color="flux", labels={"date": ""},
                                    title="LOS ANGELES : IMPORTATIONS, EXPORTATIONS, VIDES")), width="stretch")
        an_dernier = pola[pola["date"] > pola["date"].max() - pd.DateOffset(months=12)]
        t = an_dernier["total"].sum()
        pi, pe, pv = (an_dernier[c].sum() / t * 100 for c in ("imports", "exports", "vides"))
        lecture(f"Sur 12 mois : importations {pi:.0f} %, exportations {pe:.0f} %, vides {pv:.0f} % des EVP. "
                f"Los Angeles importe {pi/pe:.1f} fois plus qu'il n'exporte : les conteneurs arrivent pleins "
                "d'Asie et repartent souvent vides. C'est ce déséquilibre qui explique la part élevée des vides.")

    if prevision is not None:
        with st.container(border=True, key="carte_3"):
            fig = go.Figure()
            fig.add_scatter(x=serie.index, y=serie.values, mode="lines", name="Réel", line=dict(color=VIOLET))
            fig.add_scatter(x=prevision["date"], y=prevision["haut"], mode="lines", line=dict(width=0), showlegend=False)
            fig.add_scatter(x=prevision["date"], y=prevision["bas"], mode="lines", fill="tonexty", line=dict(width=0),
                            name="Fourchette (± MAE)", fillcolor="rgba(232,74,138,0.20)")
            fig.add_scatter(x=prevision["date"], y=prevision["prevu"], mode="lines", name="Prévision", line=dict(color=ROSE, width=3))
            fig.add_scatter(x=anomalies.index, y=anomalies.values, mode="markers", name="Mois anormal",
                            marker=dict(color=VIOLET_FONCE, size=11, symbol="x"))
            fig.update_layout(title=f"PRÉVISION 6 MOIS ET MOIS ANORMAUX · {perimetre.upper()}", yaxis_title="EVP")
            st.plotly_chart(joli(fig, 380), width="stretch")
            meilleur = "fait mieux que" if mae < mae_reference else "ne fait pas mieux que"
            recent = f" Le plus récent est {anomalies.index[-1]:%m/%Y}." if len(anomalies) else ""
            lecture(f"Le modèle prévoit entre {prevision['prevu'].min()/1e6:.2f} et {prevision['prevu'].max()/1e6:.2f} M EVP "
                    f"par mois jusqu'en {prevision['date'].iloc[-1]:%m/%Y}, à ± {mae/1e3:.0f} k EVP près. Il {meilleur} la "
                    f"référence « même mois l'an dernier » ({mae/1e3:.0f} k contre {mae_reference/1e3:.0f} k EVP d'erreur). "
                    f"{len(anomalies)} mois anormaux sont détectés depuis 2019.{recent} Ces écarts coïncident souvent avec des chocs "
                    "(rattrapage post-Covid, anticipation de droits de douane) que le modèle ne peut pas prévoir : "
                    "leur cause est à vérifier avant de planifier.")
            if mae >= mae_reference:
                lecture("Le modèle n'est pas plus précis que la référence simple : la fourchette reste utile pour "
                        "dimensionner les moyens, mais il faudrait l'améliorer (variables externes, autre modèle).")
            st.caption(f"Erreur moyenne du modèle (MAE, 12 derniers mois) : {mae:,.0f} EVP par mois. "
                       f"Référence « même mois l'an dernier » : {mae_reference:,.0f} EVP. "
                       f"Un mois est anormal si l'écart au modèle dépasse {SEUIL_ANOMALIE} écarts-types.")
            if mois_manquants:
                st.caption("Mois absents de la source, comblés par interpolation : "
                           + ", ".join(f"{m:%m/%Y}" for m in mois_manquants) + ".")

# ---------------------------------------------------------------- 8. Comparaison mondiale
section("Comparaison mondiale : CPPI et CNUCED")
st.markdown('<div class="note">Les deux sources ne mesurent pas la même chose : le CPPI note des <b>ports</b> '
            "(temps passé par les navires, rang 1 = le plus efficace) ; la CNUCED donne des chiffres par <b>pays</b>. "
            "On les lit côte à côte, sans les combiner.</div>", unsafe_allow_html=True)
g, m, d = st.columns(3)
with g.container(border=True, key="carte_4"):
    if cppi.empty:
        st.warning("CPPI indisponible.")
    else:
        fig = px.bar(cppi, x="rang", y="port", color=cppi["edition"].astype(str), barmode="group", orientation="h", text="rang",
                     title="RANG CPPI PAR ÉDITION (1 = MEILLEUR)", labels={"color": "Édition", "port": ""})
        fig.update_yaxes(categoryorder="total descending")
        st.plotly_chart(joli(fig, 360), width="stretch")
        dern = cppi["edition"].max()
        r = cppi[cppi["edition"] == dern].set_index("port")["rang"]
        r0 = cppi[cppi["edition"] == dern - 1].set_index("port")["rang"]
        lecture(f"En {dern}, Los Angeles est {int(r['Los Angeles'])}e et Long Beach {int(r['Long Beach'])}e : le complexe "
                f"reste dans le dernier quart du classement, malgré un gain de {int(r0['Los Angeles'] - r['Los Angeles'])} "
                f"places pour Los Angeles. Djibouti progresse aussi ({int(r0['Djibouti'])}e → {int(r['Djibouti'])}e). "
                "Les ports les plus efficaces sont des hubs de transbordement, comme Yangshan et Tanger Med.")
with m.container(border=True, key="carte_5"):
    if cnuced.empty:
        st.warning("CNUCED indisponible.")
    else:
        an = cnuced["annee"].max()
        tps = cnuced[(cnuced["annee"] == an) & cnuced["temps_median_jours"].notna()].sort_values("temps_median_jours")
        fig = px.bar(tps, x="temps_median_jours", y="pays", orientation="h", text_auto=".2f",
                     title=f"TEMPS MÉDIAN AU PORT, JOURS ({an})", labels={"temps_median_jours": "jours", "pays": ""})
        fig.update_traces(marker_color=[ROSE if p == "États-Unis" else MAUVE for p in tps["pays"]])
        st.plotly_chart(joli(fig, 360), width="stretch")
        t_us = tps.set_index("pays").loc["États-Unis", "temps_median_jours"]
        t_mo = tps.set_index("pays").loc["Monde", "temps_median_jours"]
        lecture(f"Aux États-Unis, un porte-conteneurs reste en médiane {t_us:.2f} jour au port, soit "
                f"{(t_us/t_mo - 1)*100:+.0f} % par rapport à la moyenne mondiale ({t_mo:.2f} jour). "
                "C'est cohérent avec le mauvais rang CPPI de Los Angeles et Long Beach.")
with d.container(border=True, key="carte_6"):
    if not cnuced.empty:
        esc = cnuced[(cnuced["pays"] != "Monde")]
        fig = px.bar(esc, x="escales", y="pays", color=esc["annee"].astype(str), barmode="group", orientation="h", text_auto=",.0f",
                     title="ESCALES DE PORTE-CONTENEURS", labels={"color": "Année", "pays": ""})
        st.plotly_chart(joli(fig, 360), width="stretch")
        e = cnuced.pivot(index="pays", columns="annee", values="escales")
        a1, a0 = e.columns.max(), e.columns.max() - 1
        ea = "Afrique de l'Est (région de Djibouti)"
        lecture(f"Entre {a0} et {a1}, les escales en Afrique de l'Est passent de {e.loc[ea, a0]:.0f} à "
                f"{e.loc[ea, a1]:.0f} ({(e.loc[ea, a1]/e.loc[ea, a0] - 1)*100:+.0f} %), contre "
                f"{(e.loc['États-Unis', a1]/e.loc['États-Unis', a0] - 1)*100:+.0f} % aux États-Unis. Cette hausse est "
                "cohérente avec la crise de la mer Rouge, qui a reporté du transbordement vers Djibouti.")

# ---------------------------------------------------------------- 9. En direct
section("En direct : navires et météo du port")
g, d = st.columns([3, 2])
with g.container(border=True, key="carte_7"):
    if ais.empty or "lat" not in ais.columns:
        st.warning("Carte AIS indisponible : couverture gratuite faible ou clé non configurée.")
    else:
        # On garde seulement les positions valides dans la zone du port (l'AIS envoie parfois 91 / 181 = inconnu)
        zone = ais[ais["lat"].between(33.55, 33.80) & ais["lon"].between(-118.45, -118.05)]
        fig = px.scatter_map(zone, lat="lat", lon="lon", color="statut", hover_name="nom",
                             hover_data=["type", "vitesse_noeuds"], center={"lat": 33.72, "lon": -118.24},
                             zoom=10.3, map_style="open-street-map",
                             title=f"NAVIRES AUTOUR DU PORT · {len(zone)} navires · {ais['capture_utc'].iloc[0]}")
        fig.update_traces(marker=dict(size=13, opacity=0.9))
        st.plotly_chart(joli(fig, 430), width="stretch")
        if len(zone):
            parts = zone["statut"].value_counts()
            detail_statut = ", ".join(f"{n} {s.lower()}" for s, n in parts.items())
            lecture(f"{len(zone)} navires visibles au moment de la capture : {detail_statut}. "
                    "Un nombre élevé de navires au mouillage signalerait une file d'attente avant quai. "
                    "La couverture AIS gratuite est partielle : ce chiffre est un minimum.")
with d.container(border=True, key="carte_8"):
    if meteo is None:
        st.warning("Météo indisponible pour le moment.")
    else:
        fig = px.line(meteo_h, x="heure", y=["rafales", "vent"], title="VENT PRÉVU SUR 72 H (km/h)",
                      labels={"heure": "", "value": "km/h"})
        fig.add_hline(y=RAFALE_ALERTE, line_dash="dash", line_color=ROSE, annotation_text="alerte")
        st.plotly_chart(joli(fig, 205), width="stretch")
        fig = px.line(meteo_h, x="heure", y="vagues", title="HAUTEUR DES VAGUES (m)", labels={"heure": ""})
        fig.add_hline(y=VAGUE_ALERTE, line_dash="dash", line_color=ROSE, annotation_text="alerte")
        st.plotly_chart(joli(fig, 205), width="stretch")
        h_raf = meteo_h.loc[meteo_h["rafales"].idxmax()]
        h_vag = meteo_h.loc[meteo_h["vagues"].idxmax()]
        etat_raf = "au-dessus" if h_raf["rafales"] >= RAFALE_VIGILANCE else "en dessous"
        etat_vag = "au-dessus" if h_vag["vagues"] >= VAGUE_ALERTE else "en dessous"
        lecture(f"Rafales maximales de {h_raf['rafales']:.0f} km/h le {h_raf['heure']:%d/%m à %Hh} ({etat_raf} du seuil de "
                f"vigilance) ; vagues jusqu'à {h_vag['vagues']:.1f} m le {h_vag['heure']:%d/%m à %Hh} ({etat_vag} du seuil "
                "d'alerte). Ces pics guident le calendrier des entrées de navires.")

# ---------------------------------------------------------------- 10. Recommandations
section("Recommandations d'action")
recos = []
if prevision is not None:
    haut = prevision["prevu"].max()
    if haut > serie[-12:].max():
        recos.append(f"**Anticiper un pic** : la prévision des 6 prochains mois ({haut/1e6:.2f} M EVP) dépasse "
                     "le plus haut des 12 derniers mois. Préparer équipes et créneaux camions.")
    else:
        recos.append("**Volumes stables attendus** : pas de pic au-dessus des 12 derniers mois. "
                     "Planifier les maintenances lourdes sur les mois les plus bas de la prévision.")
    if len(anomalies) and anomalies.index[-1] >= serie.index[-6]:
        recos.append(f"**Analyser le mois anormal de {anomalies.index[-1]:%m/%Y}** : en chercher la cause "
                     "(droits de douane, congestion, calendrier) avant de s'en servir pour planifier.")
if not pola.empty and part.iloc[-1] > moyenne + 2:
    recos.append(f"**Déséquilibre des flux** : {part.iloc[-1]:.0f} % de vides contre {moyenne:.0f} % en 2019. "
                 "Prévoir de l'espace et des navettes de repositionnement des vides.")
if meteo is not None and (meteo_h["rafales"][:48].max() >= RAFALE_VIGILANCE or meteo_h["vagues"][:48].max() >= VAGUE_ALERTE):
    recos.append("**Météo** : vent ou houle au-dessus du seuil dans les 48 h. Prévenir terminaux et pilotage, "
                 "espacer les entrées de navires.")
if not recos:
    recos.append("Aucune action particulière : tous les indicateurs sont dans leurs références.")
with st.container(border=True, key="carte_9"):
    for r in recos:
        st.markdown(f"- {r}")
    st.caption("Niveau 3 d'automatisation sur 5 : le système recommande, le responsable de la planification valide ou refuse.")

# ---------------------------------------------------------------- 11. Méthode, limites et traçabilité
with st.expander("Méthode et limites"):
    st.markdown(
        "- **Prévision** : lissage exponentiel de Holt-Winters (tendance + saisonnalité de 12 mois). "
        "Fourchette = prévision ± erreur moyenne mesurée sur les 12 derniers mois.\n"
        "- **Anomalie** : écart entre le réel et le modèle supérieur à 2 écarts-types.\n"
        "- **Limites** : Long Beach inaccessible depuis Djibouti (Access Denied le 18/11/2026), d'où des volumes "
        "sur Los Angeles seul ; chocs (Covid, congestion 2021-2022, droits de douane 2025) non prévisibles ; "
        "seuils météo à valider avec la capitainerie ; couverture AIS gratuite partielle ; la CNUCED ne publie "
        "pas Djibouti seul et s'arrête en 2023 ; le rang CPPI 2023 de Djibouti a été contesté par le pays.")

def etat_source(ligne):
    # Une source inaccessible est signalée comme telle, avec la date du constat
    if "INACCESSIBLE" in str(ligne["remarque"]):
        return f"<b>{ligne['source']}</b> : <span style='color:{ROSE}'>inaccessible (constaté le {ligne['date_recuperation']})</span>"
    return f"<b>{ligne['source']}</b> : {ligne['date_recuperation']}"


credit = ""
if os.path.exists(CREDIT_BANDEAU):
    credit = "<br>Photo du bandeau : " + open(CREDIT_BANDEAU, encoding="utf-8").read().strip()

lignes = " · ".join(etat_source(l) for _, l in sources.iterrows()) \
    if not sources.empty else "fichier data/sources.csv manquant"
st.markdown(f'<div class="trace"><b>TRAÇABILITÉ DES DONNÉES</b> — {lignes}<br>'
            f'Page générée le {datetime.now(timezone.utc):%d/%m/%Y à %H:%M} UTC · '
            f'adresses complètes dans data/sources.csv et le README.{credit}</div>', unsafe_allow_html=True)
