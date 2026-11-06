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
