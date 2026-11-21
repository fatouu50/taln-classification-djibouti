# Le projet expliqué de A à Z

Ce document explique **ce qu'on fait, pourquoi, et comment ça marche**, étape par étape, sans boîte noire.

---

## 1. La question

> Peut-on deviner automatiquement le **thème** d'un article de presse djiboutien en lisant **seulement son titre** ?

Six thèmes possibles : `politique`, `economie`, `education`, `sante`, `sport`, `societe`.

C'est une tâche de **classification de texte** : on donne un texte à la machine, et elle choisit une étiquette parmi six.

Exemple :

| Titre | Thème attendu |
|---|---|
| Djibouti lance une vaste campagne de vaccination contre la poliomyélite | sante |
| Le chef de l'Etat préside les travaux de la 15e séance du Conseil des ministres | politique |
| Les Requins de Djibouti sans complexe face aux Étalons du Burkina Faso | sport |

---

## 2. Les données : d'où viennent les titres

**Sources** : deux sites de presse publics, **La Nation** et l'**ADI** (Agence Djiboutienne d'Information).

**Pourquoi ces sites ?** Ils rangent leurs articles par **rubrique** (Santé, Sport, Économie…). On peut donc récupérer des titres **déjà classés** par le journal lui-même.

**Comment on les a récupérés** : avec le script `collecter_corpus.py`.

1. Il ouvre chaque rubrique (par exemple `lanation.dj/category/sante/`).
2. Il lit les 4 premières pages de cette rubrique.
3. Sur chaque page, il récupère le titre et le lien de chaque article.
4. Il note la rubrique, qui devient le thème proposé (`theme_rubrique`).
5. Il supprime les doublons et enregistre le tout dans `data/corpus_brut.csv`.

On appelle ça du **web scraping** : un programme lit les pages web à la place d'un humain.

**Résultat** : 803 titres collectés en une minute.

---

## 3. Le nettoyage : retirer ce qui fausse tout

Les titres de l'ADI arrivaient « sales » :

```
03 déc.GOUVERNEMENTLe chef de l'Etat préside les travaux…
```

La date et **le nom de la rubrique** étaient collés devant le titre.

**Pourquoi c'est grave** : si le mot « GOUVERNEMENT » ou « SANTÉ » reste dans le texte, le modèle n'a plus besoin de comprendre le titre : il lit la réponse écrite dedans. Il aurait des scores excellents, mais complètement faux. C'est ce qu'on appelle une **fuite d'information** (*data leakage*).

**Ce qu'on a fait** (dans `notebooks/01_corpus.ipynb`) :
- des règles qui enlèvent la date et la rubrique en début de titre ;
- l'exclusion de 98 titres dont le texte contenait tout le début de l'article.

**Résultat** : 699 titres propres.

---

## 4. L'échantillon et la vérification humaine

Le sujet demande **150 à 300 titres annotés**. On en a tiré **240**, soit **40 par thème**, au hasard (`random_state=42`, pour que le tirage soit reproductible).

**Pourquoi 40 par thème ?** Pour que chaque thème soit aussi représenté que les autres. Sinon, le modèle favoriserait les thèmes les plus nombreux.

**Pourquoi vérifier à la main ?** La rubrique du site n'est pas toujours juste. Exemple : « Course du Sud : À Mouloud, quatre distances… » était rangé en *politique* alors que c'est du *sport*.

**Comment** : Fatouma a relu les titres 1 à 120, Madina les titres 121 à 240. Pour chaque titre, on a gardé ou corrigé le thème (`theme_final`), ou exclu le titre s'il était inutilisable.

**Résultat** :
- 236 titres gardés, 4 exclus ;
- **14 thèmes corrigés sur 236 (5,9 %)** : la rubrique du site est une étiquette fiable à environ 94 % ;
- fichier final : `data/corpus.csv`.

---

## 5. Transformer un texte en nombres : le TF-IDF

Un modèle ne comprend pas les mots, seulement les **nombres**. Il faut donc transformer chaque titre en une liste de nombres. C'est la **représentation**.

**Étape 1, le vocabulaire** : on liste tous les mots qui apparaissent dans les titres (par exemple 1 500 mots différents).

**Étape 2, un vecteur par titre** : chaque titre devient une liste de 1 500 nombres, un par mot du vocabulaire. Presque tous valent 0, car un titre ne contient qu'une dizaine de mots.

**Étape 3, le poids de chaque mot (TF-IDF)** :
- **TF** (*Term Frequency*) : combien de fois le mot apparaît **dans ce titre** ;
- **IDF** (*Inverse Document Frequency*) : le mot est-il **rare dans l'ensemble des titres** ?
- **TF-IDF = TF × IDF** : un mot pèse lourd s'il est présent dans ce titre **et** rare ailleurs.

Exemple :
- « paludisme » : rare dans le corpus, très lié à la santé, donc **poids élevé** ;
- « Djibouti » : présent partout, n'aide à reconnaître aucun thème, donc **poids faible** ;
- « de », « la » : présents partout, donc **poids presque nul**.

---

## 6. Le modèle : la régression logistique

La **régression logistique** apprend, pour chaque thème, un **poids pour chaque mot**.

- Pour le thème *sante* : « vaccination » +, « paludisme » +, « hôpital » +, « match » −
- Pour le thème *sport* : « football » +, « médaille » +, « athlète » +

Pour un nouveau titre, elle additionne les poids de ses mots pour chaque thème, transforme ces scores en **probabilités**, et choisit le thème le plus probable.

```
« Lutte contre le paludisme : lancement des opérations d'épandage »
→ sante 0,81 · societe 0,07 · politique 0,05 · …  →  sante
```

**Pourquoi ce modèle ?** Il est simple, rapide et **explicable** : on peut afficher les mots qui ont le plus de poids pour chaque thème. C'est notre **baseline**.

---

## 7. La baseline : pourquoi commencer simple

Une **baseline** est un modèle de référence simple. Le sujet l'impose : **TF-IDF sur les mots + régression logistique**.

Elle sert à répondre à une question : *un modèle plus sophistiqué fait-il vraiment mieux ?* Sans baseline, on ne peut pas savoir si une méthode plus complexe vaut l'effort.

Dans le code, le TF-IDF et la régression logistique sont collés dans un **pipeline** :

```python
baseline = make_pipeline(TfidfVectorizer(lowercase=True), LogisticRegression(max_iter=1000))
```

---

## 8. L'évaluation : la validation croisée

**Le piège** : si on teste le modèle sur les titres qu'il a appris, il aura un score parfait. Ce score ne veut rien dire, comme un élève qui a vu le sujet de l'examen à l'avance.

**La solution, la validation croisée en 5 plis** :

1. On découpe les 236 titres en **5 parts** d'environ 47 titres.
2. On entraîne sur 4 parts et on teste sur la 5e.
3. On recommence 5 fois, en changeant la part de test.
4. Au final, **chaque titre a été prédit par un modèle qui ne l'avait jamais vu**.

```
Pli 1 : [TEST ][train][train][train][train]
Pli 2 : [train][TEST ][train][train][train]
Pli 3 : [train][train][TEST ][train][train]
Pli 4 : [train][train][train][TEST ][train]
Pli 5 : [train][train][train][train][TEST ]
```

- **Stratifiée** : chaque part garde la même proportion de chaque thème.
- **Pas de fuite** : grâce au pipeline, le TF-IDF est réappris à chaque pli **sur les titres d'entraînement seulement**.

**Pourquoi pas un simple découpage 80 % / 20 % ?** Avec 236 titres, le test ne compterait qu'environ 47 titres, et le score dépendrait beaucoup du hasard du découpage. La validation croisée utilise tous les titres pour le test, donc le score est plus fiable.

---

## 9. La métrique : la macro-F1

Pour chaque thème, on calcule trois chiffres :

- **Précision** : quand le modèle dit « santé », a-t-il raison ? *(parmi les titres prédits santé, combien le sont vraiment)*
- **Rappel** : parmi tous les vrais titres de santé, combien le modèle en a-t-il trouvé ?
- **F1** : un compromis entre les deux (leur moyenne harmonique). Il est élevé seulement si la précision **et** le rappel sont bons.

**Macro-F1** = la moyenne simple des F1 des 6 thèmes. Chaque thème compte autant, même s'il est un peu plus petit.

**Pourquoi pas l'accuracy** (le pourcentage de bonnes réponses) ? Si un thème était rare et que le modèle l'ignorait complètement, l'accuracy resterait bonne. La macro-F1, elle, le révèle.

Lecture : 1,0 = parfait ; 0,17 = au hasard sur 6 thèmes.

---

## 10. La matrice de confusion

C'est un tableau qui croise le **vrai thème** (lignes) et le **thème prédit** (colonnes).

- La **diagonale** donne les bonnes réponses.
- **Hors diagonale**, on voit les erreurs, et surtout **quels thèmes sont confondus**.

Exemple de lecture : si la case (ligne `economie`, colonne `politique`) vaut 6, c'est que 6 titres d'économie ont été pris pour de la politique. C'est logique : « Le ministre Dawaleh reçoit une délégation du FMI » parle d'économie, mais avec un vocabulaire politique (ministre, délégation, reçoit).

---

## 11. L'approche 2 : le TF-IDF sur les caractères

Au lieu de découper en **mots**, on découpe en **morceaux de lettres** (*n-grammes de caractères*), par exemple de 2 à 5 lettres.

```
« vaccination » → « va », « vac », « acci », « ccina », « ation »…
```

**Pourquoi ça peut marcher mieux sur nos titres** :
- **Les variantes d'un même mot** sont rapprochées : « vaccin », « vaccination », « vacciner » partagent « vacc ».
- **Les fautes de frappe et les accents** pèsent moins : « Education » et « Éducation » partagent « ducation ».
- **Les mots rares**, comme les noms propres et les sigles (MENFOP, CNSS, INSP), sont mieux exploités.
- **Les titres sont courts** : une dizaine de mots seulement, donc chaque indice compte.

On garde la même régression logistique et la même validation croisée. Seule la représentation change, et la comparaison est donc juste.

---

## 12. Le tableau comparatif

| Approche | Représentation | Modèle | Macro-F1 |
|---|---|---|---|
| Baseline | TF-IDF mots | Régression logistique | 0,589 |
| Approche 2 | TF-IDF caractères (2 à 5) | Régression logistique | 0,643 |

On complète aussi le F1 de chaque thème, pour voir **où** l'approche 2 gagne ou perd.

---

## 13. L'analyse des erreurs

Le sujet demande **au moins 10 erreurs analysées**. Pour chacune, on note :
- le titre ;
- le vrai thème et le thème prédit ;
- **pourquoi** le modèle s'est trompé.

Les causes probables :
- **Thèmes qui se chevauchent** : « Le Parlement adopte le budget 2026 » (politique ou économie ?).
- **Vocabulaire trompeur** : « ministre » apparaît dans presque tous les thèmes.
- **Titre coupé ou trop court** : « Il était une fois le CCFAR… ».
- **Biais de source** : presque tous les titres d'éducation viennent de La Nation (40 sur 43), presque tous ceux de société de l'ADI (39 sur 42). Le modèle apprend en partie le **style d'un journal** plutôt que le thème.

---

## 14. La démonstration

Une petite application **Streamlit** :
1. on colle un titre ou un article ;
2. le modèle affiche le thème prédit et la probabilité de chaque thème.

Le modèle est entraîné une fois sur les 236 titres, puis enregistré et rechargé par l'application.

---

## 15. Les limites (à dire au jury avant qu'il ne les pose)

- **Petit corpus** : 236 titres, donc les scores ont une marge d'incertitude.
- **Étiquettes issues des rubriques**, vérifiées par une seule personne par titre : pas de mesure d'accord entre annotatrices.
- **Biais de source** : éducation vient presque entièrement de La Nation, société presque entièrement de l'ADI.
- **Titres seulement** : le texte de l'article apporterait plus d'information.
- **Six thèmes qui se chevauchent** : un article peut être à la fois politique et économique, mais on n'en garde qu'un.

**Perspectives** : plus de titres, une double annotation avec un kappa de Cohen, le texte complet des articles, un Transformer multilingue (comme CamemBERT pour le français), une classification multi-étiquettes.

---

## Le chemin en une ligne

```
Sites de presse → collecte (803) → nettoyage (699) → échantillon (240) → vérification (236)
→ TF-IDF → régression logistique → validation croisée → macro-F1 + matrice de confusion
→ comparaison mots / caractères → analyse des erreurs → démonstration
```
