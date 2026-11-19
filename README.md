# Classification thématique d'articles djiboutiens

Projet TALN — Master 2 IAMD, Université de Djibouti (novembre 2026), binôme 1.
Enseignant : Dr Houssein Ahmed Assowe.

**Question :** peut-on classer automatiquement des titres d'articles djiboutiens en six thèmes
(politique, économie, éducation, santé, sport, société) ?

## Organisation du dépôt

| Dossier / fichier | Contenu |
| --- | --- |
| `data/corpus.csv` | Titres annotés (source, lien, date, étiquettes des deux annotateurs, étiquette finale) |
| `data/guide_annotation.md` | Définition des thèmes et règles d'annotation |
| `notebooks/` | Baseline, approche 2, évaluation, analyse d'erreurs |
| `app.py` | Interface : coller un titre et obtenir son thème |
| `resultats/` | Tableau comparatif, matrices de confusion |
| `rapport/` | Rapport et diapositives |

## Installation

```bash
pip install -r requirements.txt
```

## Avancement

- [ ] Corpus annoté et double annotation
- [ ] Baseline : TF-IDF mots + régression logistique
- [ ] Approche 2
- [ ] Évaluation : macro-F1, matrice de confusion
- [ ] Analyse d'au moins 10 erreurs
- [ ] Interface de démonstration
- [ ] Rapport et diapositives
