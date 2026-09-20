# Evaluation method

The public experiment asks whether an implemented possession model can learn a synthetic lineup signal and produce auditable comparisons. It does not ask whether an NBA player ranking is correct.

Three models receive the same split and standardized context. The context-only model ignores player identities. The additive one-hot baseline sums offensive class effects and subtracts defensive class effects, then adds a linear context term. ThetaNN uses mean-pooled player embeddings, an offensive-minus-defensive intercept channel, season/context inputs, and six masked heads. The two baselines optimize points cross-entropy; ThetaNN also optimizes the five auxiliary heads and small embedding/intercept penalties.

All models train for up to 100 epochs, with Adam at learning rate 0.02. The checkpoint with lowest validation points negative log likelihood is restored. Training is CPU-only, uses one thread, and enables deterministic PyTorch algorithms. Fixture seed is 17; initialization seed is 43. There is no benchmark-driven hyperparameter search in this release.

On held-out games, the primary metric is the average of each game's mean negative log likelihood (NLL, natural log units). Lower is better. A possession-weighted version is also saved; it is equal here because all games have 48 possessions. Expected-points mean squared error is a secondary check on the numerical expectation.

For each baseline, a paired bootstrap samples the same 12 test games with replacement 2,000 times, using seed 91. The difference is ThetaNN NLL minus baseline NLL. The synthetic comparison gate passes only when the entire 90% interval is below zero for both baselines. This interval is conditional on the already-fitted models and this one fixture; it does not include training-seed variation or support a broad real-world performance claim.

The contribution scorer uses the selected ThetaNN checkpoint. For each player's observed test possessions, it substitutes every eligible same-season player who is absent from both current lineups. It averages the expected-points change over replacements and then over possessions, multiplies by 100, and centers each side using the corresponding appearance counts. Defense uses the opposite sign so positive defense means fewer predicted opponent points. Fewer than 50 scored possessions on either side yields `low_sample`; no support on either side yields `no_support`.

The comparison is descriptive: players were not randomly assigned to real lineups, replacements can be unrealistic, and correlated teammates can make credit underdetermined. Centering fixes a reporting convention, not identifiability. No player-level confidence interval is claimed. The synthetic players' names are only identifiers and should never be interpreted as real athletes.

Tests protect the decisions that can silently invalidate an analysis: whole-game splitting, training-only scaling/vocabulary, label reconciliation, unique lineups, unknown-player behavior, order-invariant pooling, empty event masks, a live intercept path, paired-bootstrap sign, and complete report generation. CI runs these checks and the full public demo. The checked-in report remains an inspectable reference; CI uploads each newly computed report rather than silently rewriting it.
