# collecter_corpus.py
# Collecte automatique des titres d'articles de La Nation et de l'ADI, rubrique par rubrique.
# Résultat : data/corpus_brut.csv (titre, source, lien, date, rubrique du site, thème proposé par la rubrique)
# Lancer :  python collecter_corpus.py

import re
import time
from datetime import date

import pandas as pd
import requests
from bs4 import BeautifulSoup

PAGES = 4            # nombre de pages lues par rubrique
PAUSE = 1            # secondes d'attente entre deux pages, pour ne pas surcharger les sites
ENTETE = {"User-Agent": "Mozilla/5.0 (projet etudiant TALN, Universite de Djibouti)"}

# Rubrique du site -> thème du projet
LA_NATION = {
    "politique": "politique",
    "economie": "economie",
    "education": "education",
    "sante": "sante",
    "sports": "sport",
    "societe-culture": "societe",
}
ADI = {
    "GOUVERNEMENT": "politique",
    "PARLEMENT": "politique",
    "ECONOMIE": "economie",
    "SANTE": "sante",
    "SPORT": "sport",
    "CULTURE": "societe",
}


def lire_page(url):
    # Télécharge une page et renvoie son contenu HTML analysé (ou None si la page n'existe pas)
    reponse = requests.get(url, headers=ENTETE, timeout=30)
    if reponse.status_code != 200:
        print("   page indisponible :", url, reponse.status_code)
        return None
    return BeautifulSoup(reponse.text, "html.parser")


lignes = []

# ---------------------------------------------------------------- 1. La Nation
for rubrique, theme in LA_NATION.items():
    for page in range(1, PAGES + 1):
        url = f"https://www.lanation.dj/category/{rubrique}/" + (f"page/{page}/" if page > 1 else "")
        html = lire_page(url)
        if html is None:
            break
        # Sur La Nation, chaque titre d'article est un lien dans une balise h3 (ou h2) de classe "entry-title"
        for titre in html.select(".entry-title a"):
            lignes.append({"titre": titre.get_text(strip=True), "source": "La Nation",
                           "url": titre["href"], "date": "", "rubrique_site": rubrique, "theme_rubrique": theme})
        print("La Nation", rubrique, "page", page, "->", len(lignes), "titres au total")
        time.sleep(PAUSE)

# ---------------------------------------------------------------- 2. ADI
for rubrique, theme in ADI.items():
    for page in range(1, PAGES + 1):
        url = f"https://www.adi.dj/category/{rubrique}" + (f"?page={page}" if page > 1 else "")
        html = lire_page(url)
        if html is None:
            break
        # Sur l'ADI, chaque article est une carte ; son titre est le lien le plus long vers /article/...
        for lien in html.find_all("a", href=re.compile(r"/article/\d+")):
            texte = lien.get_text(strip=True)
            if len(texte) < 25 or texte.upper().startswith("LIRE LA SUITE"):
                continue   # on ignore les liens « Lire la suite » et les images
            # La date est écrite au début du résumé : « DJIBOUTI, 7 décembre 2025 (ADI) »
            carte = lien.find_parent(["article", "div"])
            resume = carte.get_text(" ", strip=True) if carte else ""
            trouve = re.search(r"(\d{1,2}\s+\w+\s+20\d\d)", resume)
            lignes.append({"titre": texte, "source": "ADI", "url": requests.compat.urljoin(url, lien["href"]),
                           "date": trouve.group(1) if trouve else "", "rubrique_site": rubrique,
                           "theme_rubrique": theme})
        print("ADI", rubrique, "page", page, "->", len(lignes), "titres au total")
        time.sleep(PAUSE)

# ---------------------------------------------------------------- 3. Nettoyage et enregistrement
corpus = pd.DataFrame(lignes)
corpus = corpus.drop_duplicates(subset="url")          # un même article vu deux fois
corpus = corpus.drop_duplicates(subset="titre")        # un même titre publié deux fois
corpus["date_collecte"] = date.today().isoformat()
corpus.to_csv("data/corpus_brut.csv", index=False)

print()
print(len(corpus), "titres uniques enregistrés dans data/corpus_brut.csv")
print(corpus.groupby(["theme_rubrique", "source"]).size().unstack(fill_value=0))
