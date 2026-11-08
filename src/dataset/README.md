# SceneDataset

Depuis la racine du dépôt :

```python
from torch.utils.data import DataLoader
from src.data import SceneDataset, collate_scenes
from src.losses.scene_loss import SceneLoss

dataset = SceneDataset("dataset_preview", split="train")
loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_scenes)
criterion = SceneLoss()

for batch in loader:
    predictions = model(batch["images"].to(device))
    losses = criterion(predictions, batch["targets"])
    losses["total"].backward()
```

`model` et `device` sont définis par le code d'entraînement. La loss transfère
les cibles sur le device des prédictions.

Un échantillon contient `images` de forme `(V, 3, H, W)`, `objects` avec
`classes` (int64, `(N,)`), `positions` (float32, `(N, 3)`),
`rotations` (float32, `(N, 3, 3)`), et `cameras`
avec `positions` (float32, `(V, 3)`) et `rotations` (float32, `(V, 3, 3)`).
Les images et caméras suivent l'ordre du JSON, et les positions et rotations
sont exprimées dans le repère de la caméra 0. Les scènes sans objet sont prises
en charge. Les autres annotations ne sont pas utilisées par `SceneLoss`.

Les objets ont une taille fixe : l'échelle n'est ni chargée ni prédite.
`rotations` contient les matrices `rotation_matrix` des objets dans le repère
de référence. La représentation 6D prédite par `ObjectPredictionHead` concatène
les deux premières colonnes de ces matrices, comme le générateur.

`SceneLoss` retourne `object_class`, `object_position`,
`object_rotation`, `camera_position`, `camera_rotation` et `total`. Les
rotations (objets et caméras) utilisent la distance de Frobenius au carré, sommée sur les neuf coefficients puis moyennée sur les rotations.
Les métriques `object_rotation_angle` et `camera_rotation_angle` donnent
l'angle géodésique moyen en radians ; elles sont détachées du graphe et exclues
de `total`. Pour les objets, l'angle est minimisé sur les 24 symétries du cube
ou calculé entre les axes Z non orientés des cylindres. Les sphères sont exclues
de cette moyenne, qui vaut zéro sans objet orientable. Les deux pertes
géométriques des objets utilisent les mêmes paires de l'appariement hongrois
(classe et position). Les requêtes non appariées ne contribuent pas à ces pertes.
Tous les poids valent 1 par défaut et sont configurables, par exemple
`SceneLoss(weights={"object_position": 0.5, "object_rotation": 2.0})`.
Les prédictions doivent fournir `position` `(B, Q, 3)` et `rotation` `(B, Q, 6)`.
Les cibles peuvent être les annotations JSON brutes ou les dictionnaires du
dataset ; les rotations cibles en tenseurs peuvent aussi être de forme `(N, 6)`.

Le batch contient `images` de forme `(B, V, 3, H, W)` et `targets`, une liste
de dictionnaires compatible avec `SceneLoss`, même si le nombre d'objets varie.
Les scènes d'un batch doivent avoir le même nombre de vues et la même taille
d'image.

Les images RGB sont chargées en float32 dans `[0, 1]`. Le paramètre `transform`
accepte une fonction appliquée à chaque tenseur image pour, par exemple,
effectuer la normalisation attendue par le modèle. Le prétraitement spécifique
à DINOv2 reste à configurer par l'appelant. Éviter les recadrages et retournements
qui changeraient la géométrie sans modifier les annotations.
