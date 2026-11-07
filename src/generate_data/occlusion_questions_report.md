# Compte rendu : dataset `occlusion_questions` (7 novembre 2026)

## Objectif

Remplacer `random_8_views`, dont les objets étaient trop petits et presque
jamais occultés, par un dataset destiné au VLM :

- des objets plus grands dans l'image ;
- de vraies occultations entre objets ;
- des **questions multivues** que DETR ne sait pas traiter directement.

Configuration : [occlusion_questions_config.json](occlusion_questions_config.json).
Le dataset n'est pas encore généré ; la commande est en fin de document.

## Scènes

| Paramètre | `random_8_views` | `occlusion_questions` |
|---|---|---|
| Objets par scène | 2–4 | 3–4 |
| Étendue des positions | ±3,0 m | ±1,7 m |
| Sphères / cylindres | rayon fixe de 1 m | rayon tiré, taille comparable aux cubes |
| Couleurs | RGB aléatoire | 6 couleurs nommées |
| Caméras | rayon 15 m, hauteur 10 m, 60 mm | rayon 8 m, hauteur 2,5 m, 55 mm |
| Vues | 8, azimut aléatoire, roulis ±10° | inchangé |
| Splits | 5 000 / 500 | 5 000 / 500 / 500 |

Mesures sur 600 scènes, par lancer de rayons, sans rendu :

| | Avant | Après |
|---|---|---|
| Taille apparente d'un objet (racine de sa surface) | ~35 px | ~51 px |
| Couples (objet, vue) partiellement masqués | 2 % | 21 % |
| Scènes avec un objet entièrement caché dans une vue | 0 % | 42 % |
| Couples touchant le bord de l'image | 0 % | 7,5 % |

Le critère d'occultation exige au moins 2 couples visibles à 50 % au plus. Il
était satisfait d'emblée dans environ 90 % des scènes et l'a été pour 600 scènes
sur 600 après au plus 30 nouvelles dispositions des mêmes objets. Avec 5 objets,
le placement échouait dans 10 % des scènes, d'où le maximum de 4.

## Nouveautés du générateur

- **Visibilité** (`schema_version: 2`) : pour chaque caméra et chaque objet,
  pixels visibles, pixels non occultés, fraction visible, objets occultants,
  boîte projetée. Le calcul est géométrique (BVH) et coûte environ 0,5 s par
  scène.
- **`linked_scale_axes_by_type`** : sphères rondes et cylindres à section
  circulaire, de taille variable.
- **`color_palette`** : chaque objet reçoit un `color_name`, ce qui permet de
  le désigner, par exemple « the red cube ».
- **`questions.json`** par scène, produit par [questions.py](questions.py).
- Les anciennes configurations produisent des scènes identiques au bit près
  (vérifié sur `random_8_views`, `variable_cubes` et `uniform_objects`).

## Questions

Il y a environ 16 questions par scène, en anglais, comme le préentraînement du
LLM. Les réponses appartiennent à un vocabulaire fermé et sont évaluées par
correspondance exacte.

| Famille | Compétence | Exemple |
|---|---|---|
| A | Relations vues depuis une caméra | *From camera 5, is the cube to the left or to the right of the sphere?* → right |
| B | Visibilité, occultation | *Which object blocks the view of the cyan sphere from camera 5?* → cylinder |
| D | Correspondance entre vues | *Can the rightmost object in the image from camera 1 be seen from camera 4?* → yes |
| F | Changement de perspective | *The cyan sphere cannot be seen from camera 2. From that camera, is it to the left or to the right of the red cube?* → left |
| G | Comparaisons métriques | *Around the center of the scene, what is the angle between camera 2 and camera 4?* → 90 degrees |

Garde-fous :

- les relations trop proches d'une marge de sécurité ne sont jamais posées ;
- un objet n'est cité que si une description courte le désigne sans ambiguïté ;
- pour chaque modèle de question, la réponse majoritaire est plafonnée.

Statistiques sur 600 scènes : 9 507 questions, soit 15,8 par scène. Pour
chaque modèle, la réponse majoritaire représente au plus 60 %. Par famille :
A 2 397, G 2 105, F 1 902, B 1 769, D 1 334.

## Vérifications

- Une quinzaine de réponses contrôlées à l'œil sur 3 scènes rendues : toutes justes.
- Les fractions visibles calculées correspondent aux images, y compris pour les
  objets qui ne dépassent que de 4 à 16 %.
- Les questions produites dans Blender et celles régénérées avec `questions.py`
  sont identiques.

## Limites connues

- Certains modèles sont rares, environ 0,1 par scène : `B_hidden`,
  `B_num_cameras`, `D_shared` et `G_closest_pair`. Sur 500 scènes de validation,
  leur précision reposera sur environ 50 exemples.
- `requires_multiview` est une étiquette heuristique. Seule une ablation
  monovue mesure le besoin réel de plusieurs vues.
- La diversité linguistique est faible, puisque les questions viennent de
  modèles de phrases. Le modèle ne généralisera pas à des formulations
  nouvelles.
- Les caméras ont changé : le modèle vision doit être réentraîné sur ce dataset.

## Points à respecter à l'entraînement

- La loss doit porter **uniquement sur les tokens de la réponse**.
- Rapporter la précision par famille et par modèle de question, en la comparant
  à quatre références : réponse majoritaire, LLM aveugle, VLM monovue, et
  DETR + exécuteur.

## Commandes

```bash
.venv/bin/python src/generate_data/generate_dataset.py --config src/generate_data/occlusion_questions_config.json --output dataset_occlusion_questions
python src/generate_data/questions.py --dataset dataset_occlusion_questions --stats
```
