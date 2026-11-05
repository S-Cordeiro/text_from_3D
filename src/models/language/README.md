# LLM décodeur seul

Architecture GPT inspirée de [build-nanogpt](https://github.com/karpathy/build-nanogpt).
`GPTConfig` utilise les mêmes noms : `block_size`, `vocab_size`, `n_layer`,
`n_head`, `n_embd`. Par défaut, le modèle a les dimensions de GPT-2 124M.
Il part de poids aléatoires ; aucun poids préentraîné n'est téléchargé.
Les blocs pré-normalisés réutilisent `src/models/blocks.py`, avec attention
causale pour le langage et attention bidirectionnelle pour la vision.
Le LLM utilise un feedforward SwiGLU sans biais :
`c_proj(silu(c_gate(x)) * c_fc(x))`, de dimension cachée `8 * n_embd // 3`,
pour un nombre de paramètres proche du MLP GELU initial. Ce changement rend
les anciens checkpoints GELU incompatibles avec cette architecture.
Les embeddings de tokens et la tête de sortie partagent leurs poids.

## Préparer les données

Depuis la racine du projet :

```bash
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m src.prepare_language_data \
  --train-text data/train.txt --val-text data/val.txt \
  --output-dir data/language
```

Fournir deux corpus UTF-8 distincts et sans chevauchement. Le tokenizer réutilise
le BPE GPT-2 de tiktoken (50 257 tokens), avec `encode`, `decode` et un token de
fin de texte optionnel. Le premier chargement peut télécharger les données du
tokenizer. Le script de préparation charge chaque corpus en mémoire : il est
prévu pour les premiers essais, pas pour prétraiter des milliards de tokens.
Il écrit `train_000000.npy` et `val_000000.npy` (uint16). Le chargeur accepte
plusieurs shards `train_*.npy` et `val_*.npy`, parcourus dans l'ordre puis bouclés.
Chaque shard doit contenir au moins `batch_size * block_size + 1` tokens ;
les fins de shard trop courtes sont ignorées. Les cibles sont décalées d'un token.

## Entraîner sur un GPU CUDA

```bash
.venv/bin/python -m src.train_language \
  --data-root data/language --out-dir runs/language \
  --batch-size 4 --block-size 256 --total-batch-size 4096 \
  --n-layer 6 --n-head 6 --n-embd 384 \
  --max-steps 1000 --warmup-steps 100
```

Cet exemple utilise un modèle plus petit que GPT-2. Ajuster le micro-batch
`--batch-size` à la VRAM disponible. `--total-batch-size` désigne un nombre de
tokens par mise à jour, obtenu par accumulation des gradients ; il doit être
divisible par `batch_size * block_size`. Le script utilise uniquement `cuda:0`
(pour choisir la carte physique, définir `CUDA_VISIBLE_DEVICES`), sans DDP.
AdamW, warmup puis décroissance cosinus, clipping à 1.0, autocast bfloat16
ou float16 avec GradScaler et `--compile` optionnel sont inclus.

Les logs incluent pertes train/validation, norme des gradients et débit.
Les checkpoints contiennent modèle, configuration, optimiseur, scaler et étape.
Le script ne propose pas encore de reprise d'entraînement.

## Charger et générer

```python
import torch
from src.models.language import GPT, GPTConfig
from src.models.language.tokenizer import Tokenizer

checkpoint = torch.load('runs/language/model_001000.pt', map_location='cpu', weights_only=True)
model = GPT(GPTConfig(**checkpoint['config']))
model.load_state_dict(checkpoint['model'])
model = model.to('cuda').eval()
tokenizer = Tokenizer()
idx = torch.tensor([tokenizer.encode('La scène contient')], device='cuda')
tokens = model.generate(idx, max_new_tokens=64, temperature=0.8, top_k=50)
print(tokenizer.decode(tokens[0].tolist()))
```

La génération garde les `block_size` derniers tokens quand le contexte est plein.
Il n'y a pas encore de cache KV ni de connexion à la vision : l'intégration VLM
viendra ensuite. Les modèles de vision vivent dans `src/models/vision/` ;
les imports doivent utiliser `src.models.vision.*`.
