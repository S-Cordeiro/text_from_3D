# LLM décodeur seul

Architecture GPT inspirée de [build-nanogpt](https://github.com/karpathy/build-nanogpt).
`GPTConfig` utilise les mêmes noms : `block_size`, `vocab_size`, `n_layer`,
`n_head`, `n_embd`. Par défaut : 8 couches, 8 têtes, dimension 768 et contexte
de 1 024 tokens, soit 96 057 600 paramètres.
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

Pour préparer FineWeb-Edu à plus grande échelle :

```bash
.venv/bin/python -m src.prepare_fineweb_edu --output-dir data/language
```

Le script télécharge progressivement les fichiers Parquet du sous-ensemble
officiel `sample-10BT`, conservés dans le cache Hugging Face. Il produit exactement
1 921 152 000 tokens d'entraînement (20 fois le nombre de paramètres par défaut)
et 10 000 000 tokens de validation, issus de documents distincts, avec un token
de fin de texte par document. Les shards contiennent au maximum 100 millions de
tokens uint16, soit environ 3,86 Go au total, en plus des Parquet en cache.
Le manifeste enregistre la révision source, les fichiers utilisés, les comptes
et le statut. Attendre le statut `complete` avant d'entraîner. Le dossier de
sortie doit être neuf ; la tokenisation interrompue ne reprend pas automatiquement.
Les options `--train-tokens`, `--val-tokens`, `--shard-size`, `--threads` et
`--revision` permettent d'ajuster la préparation. Avec le batch total par défaut
de 32 768 tokens, prévoir environ 58 629 étapes pour traiter ce volume une fois.

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
La reprise restaure modèle, optimiseur, scaler, RNG et position exacte dans les
shards, ainsi que le calendrier du taux d'apprentissage et la configuration :

```bash
.venv/bin/python -u -m src.train_language \
  --resume runs/mon_entrainement/last.pt --max-seconds 6900
```

Les checkpoints sont publiés atomiquement. `last.pt` pointe sur le plus récent ;
les trois derniers checkpoints numérotés sont conservés par défaut
(`--keep-checkpoints`). La sauvegarde a lieu toutes les `--save-every` étapes
ou `--save-seconds` secondes, puis à l'arrêt. SIGUSR1, SIGTERM et SIGINT
demandent un arrêt après la mise à jour en cours, suivi d'une sauvegarde.
`--max-seconds` fixe le budget de la session, sans modifier le calendrier total
d'entraînement. Prévoir une marge pour terminer une étape et écrire le checkpoint.

Sur Bentley, depuis la racine du projet, `bash scripts/submit_language_bentley.sh`
soumet quatre tâches Slurm de deux heures au maximum, dépendantes les unes des
autres. Chacune s'arrête après 1 h 55 et reprend `last.pt` à la suivante.
Un test CUDA de sauvegarde/reprise précède le premier entraînement. Le code est
figé dans le dossier du run ; les journaux Slurm sont dans `logs/language_JOBID.log`.
Les huit heures sont un maximum cumulé d'allocation, hors attente dans la file.
L'exécution est limitée à la nuit, de 18 h à 8 h, heure de Paris : Slurm reçoit
une date de début et une échéance à 8 h. Le script refuse tout démarrage en
journée, réduit son budget si nécessaire, réserve cinq minutes pour sauvegarder
et utilise un timeout indépendant pour ne pas dépasser 8 h. Les créneaux qui
ne peuvent plus tenir avant cette échéance ne démarrent pas ; ils ne sont pas
reportés automatiquement à la nuit suivante. `--night-only` applique aussi
la vérification de plage horaire lors d'une reprise manuelle.

## Charger et générer

```python
import torch
from src.models.language import GPT
from src.models.language.tokenizer import Tokenizer

model = GPT.from_checkpoint('runs/language/model_001000.pt', device='cuda')
tokenizer = Tokenizer()
idx = torch.tensor([tokenizer.encode('La scène contient')], device='cuda')
tokens = model.generate(idx, max_new_tokens=64, temperature=0.8, top_k=50)
print(tokenizer.decode(tokens[0].tolist()))
```

La génération garde les `block_size` derniers tokens quand le contexte est plein.
Il n'y a pas encore de cache KV ni de connexion à la vision : l'intégration VLM
viendra ensuite. Les modèles de vision vivent dans `src/models/vision/` ;
les imports doivent utiliser `src.models.vision.*`.
