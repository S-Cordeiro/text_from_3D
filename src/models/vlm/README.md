# VLM à préfixe visuel

`LLaVAModel` réutilise un `SceneModel` et un `GPT`. Les object queries
contextualisées (sortie de l'object decoder, avant les têtes de prédiction)
sont projetées par une couche linéaire `d_model → n_embd`, puis concaténées
avant les embeddings du texte. Le GPT applique ses positions et son attention
causale sur toute la séquence. Les têtes objets et caméras ne sont pas exécutées.

```python
from src.models.vlm import LLaVAModel

model = LLaVAModel.from_pretrained("vision.pt", "language.pt", device="cuda")
model.train()
logits, loss = model(pixel_values, idx, targets)
loss.backward()
generated = model.generate(pixel_values, idx, max_new_tokens=32)
```

`pixel_values` conserve le format du SceneModel `(B, V, 3, H, W)`.
`idx` et `targets` ont la forme `(B, T)` : comme pour le GPT seul,
`targets[:, t]` contient le token suivant `idx[:, t]`. Les logits ont la forme
`(B, T, vocab_size)` et la loss porte uniquement sur ces positions textuelles.
La valeur `-100` permet d'ignorer certaines cibles (par exemple le prompt).
Inclure un token de début dans `idx` pour prédire le premier token de réponse.

Le contexte total doit respecter `n_objects + T <= block_size`. En génération,
le préfixe visuel est calculé une fois et conservé ; le texte le plus ancien
est tronqué quand le contexte est plein. Le résultat contient le prompt et les
nouveaux tokens, sans les tokens visuels.

On peut aussi construire `LLaVAModel(vision_model, language_model)` directement.
La projection est initialisée aléatoirement et doit être entraînée sur des paires
images/texte. Aucun gel supplémentaire des paramètres n'est imposé : les règles
du modèle visuel existant sont conservées. Pour entraîner uniquement la projection :

```python
model.vision_model.requires_grad_(False)
model.language_model.requires_grad_(False)
```

Cette modification fournit l'architecture et la génération ; elle ne lance pas
d'entraînement et n'ajoute pas de jeu de données multimodal.

## Variante Flamingo

```python
from src.models.vlm import FlamingoModel

model = FlamingoModel.from_pretrained("vision.pt", "language.pt", device="cuda")
model.train()
logits, loss = model(pixel_values, idx, targets)
loss.backward()
generated = model.generate(pixel_values, idx, max_new_tokens=32)
```

`FlamingoModel(vision_model, language_model)` accepte aussi les modèles existants.
Il insère `n_layer - 1` modules entre les blocs du GPT (au moins deux blocs requis).
Chaque module applique :

1. `x = x + tanh(attn_gate) * CrossAttention(LayerNorm(x), vision, vision)` ;
2. `x = x + tanh(ff_gate) * SwiGLU(LayerNorm(x))`.

Les deux portes scalaires sont initialisées à zéro : en mode évaluation, la
sortie initiale est celle du GPT préentraîné. Les portes reçoivent des gradients
dès le premier pas ; les branches visuelles reçoivent leur signal quand les
portes s'ouvrent. Les projections clés/valeurs de chaque cross-attention
acceptent directement la dimension du modèle visuel.

Les tokens viennent de `SceneModel.encode_multiview`, avant les décodeurs objets
et caméras. Tous les tokens de toutes les vues de la scène sont accessibles à
chaque position textuelle. Il n'y a pas de resampler ni de gestion de séquences
entrelacées images/texte : il s'agit d'une variante adaptée à une scène multivue.
Le masque causal du GPT reste appliqué au texte.

Les formats d'entrée, logits et cibles sont les mêmes que pour LLaVA, mais
les tokens visuels ne consomment pas le contexte : `T <= block_size`.
La génération calcule les tokens visuels une seule fois et tronque le texte
ancien si nécessaire. Aucun gel supplémentaire n'est imposé ; pour entraîner
uniquement les nouveaux modules, geler `vision_model` et `language_model`
avec `requires_grad_(False)` comme ci-dessus. L'optimiseur doit inclure les
paramètres de `model`, et pas seulement ceux de `model.language_model`.

Vérifications sans téléchargement :
`python -m scripts.check_flamingo` et `python -m scripts.check_vlm`.
