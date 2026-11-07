# Campagne vision sur Bentley

Le point d'entrée est `scripts/submit_vision_campaign.py`. Le code exécuté doit être
copié dans `CAMPAIGN/code/` avant soumission. Le script crée un manifeste immuable
avec les empreintes du code et soumet un job Slurm exclusif sur Bentley.

Ordre prévu :

1. Reprendre le LLM existant, avec son code figé et ses états modèle, optimiseur,
   scaler, RNG et lecteur de tokens. Conserver sa cible de 58 629 étapes et son
   scheduler. La rétention passe à deux checkpoints récents, plus les logs.
2. Auditer tous les labels train/val de `data/random_8_views` : comptes réels,
   nombre d'objets, vues, résolutions, présence des images et empreintes des
   ground truths. Signaler les anciennes configurations incompatibles sans
   réécrire les résultats historiques.
3. Entraîner trois modèles depuis zéro, chacun pendant 80 époques : référence
   256/4, profondeur 256/6 et largeur 384/4. Trois classes et cinq object queries.
4. Depuis le **même meilleur checkpoint de la référence**, entraîner trois branches
   pendant 10 époques : contrôle gelé, dernier bloc DINO dégelé, deux derniers
   blocs dégelés. LR module vision 2e-5 et DINO 1e-6. La normalisation finale
   de DINO est entraînée avec ses derniers blocs. DINO reste en mode évaluation.
5. Exporter pour chaque meilleur checkpoint les prédictions, cibles, associations
   hongroises et métriques de chacune des 500 scènes de validation. Générer les
   résumés JSON et CSV.

Tous les essais utilisent la seed 1337, un batch effectif de 16, des microbatches
identiques de 4, bf16 et la même version des pertes. L'accumulation des gradients
sur microbatches ne reproduit pas exactement une loss calculée sur un batch de 16
(notamment sa normalisation de classification) ; elle est identique entre les
nouveaux essais. Ils ne doivent pas être comparés directement aux anciens runs
à deux classes. La moyenne de l'angle objet exclut correctement les sphères.
Une seule seed constitue une exploration, pas une preuve statistique.

Les jobs ont une durée maximale de deux heures. Une pause propre sauvegarde le
curseur des données, les poids, l'optimiseur, le scheduler et les RNG, puis soumet
le prochain job avec une dépendance `afterok`. Les exécutions sont limitées à
18h–8h Europe/Paris ; si nécessaire la suite est reportée à la nuit suivante.
Une erreur arrête la chaîne. Le plafond de 32 allocations (au plus 64 h réservées)
évite une resoumission infinie ; aucun job n'est soumis après achèvement.

Fichiers conservés : `manifest.json`, `dataset_audit.json`, `llm_status.json`,
`jobs.jsonl`, logs Slurm/LLM, environnement pip, télémétrie GPU toutes les 30 s,
puis pour chaque essai `config.json`, `metrics.jsonl`, `status.json`, `train.log`,
`best.pt`, `last.pt`, `best_validation_predictions.jsonl.gz`.
Les métriques incluent les erreurs objets/caméras, la validation par nombre
 d'objets, le LR, le nombre de scènes vues, le temps écoulé et le pic mémoire GPU.
Les checkpoints incluent tous les poids DINO afin de rester autonomes.

Validation locale sans téléchargement :

```bash
.venv/bin/python -m scripts.check_vision_campaign
```

Elle vérifie les gradients sélectifs de DINO, le rejet de la collision de classes
et l'identité des poids CPU entre entraînement continu et arrêt/reprise, avec
accumulation et un dernier batch incomplet.

Après entraînement, le résumé peut être régénéré sans GPU :

```bash
.venv/bin/python scripts/summarize_vision_campaign.py CAMPAIGN
```
