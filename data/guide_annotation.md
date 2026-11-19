# Guide d'annotation — classification thématique d'articles djiboutiens

## Unité annotée
Le **titre** d'un article de presse djiboutienne (La Nation, ADI), avec son lien.

## Les 6 thèmes

| Étiquette | Ce qu'elle couvre |
| --- | --- |
| `politique` | Gouvernement, Assemblée nationale, élections, diplomatie, décrets, visites officielles, coopération entre États |
| `economie` | Port, commerce, finances, budget, investissements, emploi, entreprises, infrastructures, énergie |
| `education` | École, université, examens, formation, bourses, enseignants |
| `sante` | Hôpitaux, maladies, vaccination, campagnes sanitaires, médicaments |
| `sport` | Compétitions, clubs, athlètes, fédérations, résultats sportifs |
| `societe` | Culture, religion, environnement, jeunesse, femmes, solidarité, faits de société |

## Règles
1. On annote le **thème principal du titre**, pas le contexte de l'article.
   Exemple : « Le ministre de la Santé inaugure un hôpital » → `sante`, pas `politique`.
2. Une seule étiquette par titre.
3. Chaque annotateur étiquette **seul**, sans voir l'étiquette de l'autre (colonnes `theme_A` et `theme_B`).
4. Les désaccords sont discutés à deux ; la décision va dans `theme_final`, la raison dans `remarque`.
5. Un titre trop vague pour être classé est écarté (noter « écarté » dans `remarque`).

## Mesure de qualité
Accord entre annotateurs : pourcentage d'accord et kappa de Cohen, calculés sur `theme_A` et `theme_B`.
