# Classification thématique d'articles djiboutiens

Projet TALN — Master 2 IAMD, Université de Djibouti (novembre 2026), binôme 1 : Fatouma Soubere Mahamoud et Madina.
Enseignant : Dr Houssein Ahmed Assowe (CRMN).

**Question :** peut-on classer automatiquement des titres de presse djiboutienne en six thèmes :
politique, économie, éducation, santé, sport, société ?

| Échéance | Date |
| --- | --- |
| Dépôt final | lundi 23 novembre 2026, 18 h |
| Soutenance | mardi 24 novembre 2026 (10 min de présentation, 3 min de démo, 5 min de questions) |

## Avancement

### 1. Données
- [x] Collecte automatique des titres sur La Nation et l'ADI (`collecter_corpus.py`) : 803 titres
- [x] Nettoyage des titres ADI : date et nom de rubrique retirés (fuite d'étiquette évitée)
- [x] Exclusion de 98 titres contenant le corps de l'article : 699 titres propres
- [x] Échantillon équilibré de 240 titres, 40 par thème (`data/echantillon_a_verifier.csv`)
- [x] Vérification manuelle des titres 1 à 120 (Fatouma)
- [x] Vérification manuelle des titres 121 à 240 (Madina, issue #1)
- [x] Corpus final `data/corpus.csv` : 236 titres, 14 thèmes corrigés (5,9 %)

### 2. Modèles
- [x] Baseline : TF-IDF mots + régression logistique, validation croisée stratifiée
- [x] Approche 2 : TF-IDF caractères + régression logistique
- [x] Tableau comparatif (macro-F1, F1 par thème)
- [x] Matrices de confusion (baseline et approche 2)
- [x] Analyse de 14 erreurs, regroupées en 6 causes

### 3. Démonstration et documentation
- [ ] Interface Streamlit : coller un titre ou un article, obtenir son thème
- [ ] README complet (installation, reproduction, description des données)
- [x] `requirements.txt`

### 4. Rendu
- [ ] Rapport de 6 à 8 pages
- [ ] Diapositives (8 à 10)
- [ ] Démonstration de 2 à 4 minutes répétée
- [ ] Vérification : tout s'exécute depuis un dossier vide
- [ ] Dépôt final

## Résultats

Validation croisée stratifiée en 5 plis sur 236 titres.

| Approche | Macro-F1 moyenne | Écart-type | Plis gagnés |
| --- | --- | --- | --- |
| Baseline : TF-IDF mots + régression logistique | 0,589 | 0,035 | 1 |
| Approche 2 : TF-IDF caractères (2-5) + régression logistique | **0,643** | 0,065 | **4** |

| Thème | F1 baseline | F1 approche 2 |
| --- | --- | --- |
| economie | 0,38 | 0,52 |
| education | 0,67 | 0,75 |
| politique | 0,62 | 0,70 |
| societe | 0,57 | 0,65 |
| sante | 0,76 | 0,75 |
| sport | 0,56 | 0,49 |

Gain régulier mais modeste ; le sport recule avec les caractères. Biais de source partiel : les erreurs se rabattent en partie sur le thème propre au journal d'origine. Détails : `notebooks/02_modeles.ipynb` et `resultats/`.

## Planning

| Jour | Travail prévu |
| --- | --- |
| Ven. 20 nov. | Vérification des 240 titres ; baseline TF-IDF mots + régression logistique |
| Sam. 21 nov. | Approche 2 ; tableau comparatif ; matrice de confusion ; analyse de 10 erreurs |
| Dim. 22 nov. | Interface Streamlit ; README ; début du rapport |
| Lun. 23 nov. | Rapport, diapositives, répétition de la démo ; dépôt à 18 h |
| Mar. 24 nov. | Soutenance |

## Chaîne de traitement

1. **Collecte** : `collecter_corpus.py` lit les rubriques de La Nation et de l'ADI. Chaque titre reçoit le thème de sa rubrique (`theme_rubrique`).
2. **Prétraitement** : `notebooks/01_corpus.ipynb` nettoie les titres et tire l'échantillon.
3. **Vérification** : les 240 titres sont relus à la main ; `theme_final` est corrigé si le thème de la rubrique est faux.
4. **Baseline** : TF-IDF sur les mots + régression logistique.
5. **Approche 2** : TF-IDF sur les n-grammes de caractères + régression logistique.
6. **Évaluation** : macro-F1 en validation croisée stratifiée, matrice de confusion.
7. **Analyse d'erreurs** : au moins 10 cas commentés.
8. **Démonstration** : interface où l'on colle un texte pour obtenir son thème.

## Organisation du dépôt

| Dossier / fichier | Contenu |
| --- | --- |
| `collecter_corpus.py` | Collecte des titres par rubrique sur La Nation et l'ADI |
| `data/corpus_brut.csv` | Les 803 titres collectés, avant nettoyage |
| `data/echantillon_a_verifier.csv` | Les 240 titres tirés pour la vérification manuelle |
| `data/corpus.csv` | Corpus final vérifié (236 titres gardés, 4 exclus) |
| `data/guide_annotation.md` | Définition des six thèmes et règles d'annotation |
| `notebooks/01_corpus.ipynb` | Chargement, nettoyage, exclusions, tirage de l'échantillon |
| `notebooks/02_modeles.ipynb` | Baseline, approche 2, comparaison, matrices de confusion, analyse des erreurs |
| `resultats/` | Tableau comparatif, matrices de confusion |
| `rapport/` | Rapport et diapositives |

## Reproduire

```bash
pip install -r requirements.txt
python collecter_corpus.py        # recrée data/corpus_brut.csv (les sites évoluent : le résultat peut varier)
jupyter lab                       # puis exécuter notebooks/01_corpus.ipynb
```

Les tirages aléatoires utilisent `random_state=42` : à corpus identique, l'échantillon est identique.

## Données

- **Sources** : [La Nation](https://www.lanation.dj) et l'[Agence Djiboutienne d'Information (ADI)](https://www.adi.dj), articles publics.
- **Étiquettes** : thème de la rubrique du site, vérifié à la main.
- **Correspondance des rubriques** : La Nation `politique`, `economie`, `sante`, `sports`, `education` ; ADI `GOUVERNEMENT` et `PARLEMENT` (politique), `ECONOMIE`, `SANTE`, `SPORT`, `CULTURE` (société).

## Limites connues

- Les titres d'éducation viennent tous de La Nation, ceux de société tous de l'ADI : le modèle pourrait apprendre le style d'un journal plutôt que le thème.
- Chaque titre est vérifié par une seule personne : pas de mesure d'accord inter-annotateurs.
- La colonne `date` de l'ADI n'est pas fiable ; elle n'est pas utilisée par les modèles.
- Corpus de petite taille (240 titres) et titres courts.

## Suivi

Chaque étape fait l'objet d'une issue GitHub, fermée quand l'étape est terminée.
