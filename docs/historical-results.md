# Historical development evidence

These figures summarize recorded development artifacts. They are separate from the newly executed public synthetic benchmark, and the public example cannot reproduce historical NBA results without the corresponding rights-cleared data and full evaluation environment.

On September 20, 2026, the corpus metadata was checked against the normalized possession table itself: **1,396,353 rows and 7,230 distinct games**. See the [machine-readable scope](../reports/historical-scope.json).

| Season | Possessions | Games |
|---|---:|---:|
| 2020–21 | 208,912 | 1,080 |
| 2021–22 | 235,916 | 1,230 |
| 2022–23 | 238,496 | 1,230 |
| 2023–24 | 236,419 | 1,230 |
| 2024–25 | 237,154 | 1,230 |
| 2025–26 | 239,456 | 1,230 |
| **Total** | **1,396,353** | **7,230** |

These are retained, normalized possessions, not every raw play-by-play event. Regulation-only processing excluded overtime. The broader data-quality process also recorded dropped possessions; the retained corpus size should not be described as lossless raw-data coverage.

For the recorded 2024–25 evaluation origin, the four preceding seasons contain 919,743 possessions over 4,770 games. The historical split logic holds out 10% of the last preceding season's games for early stopping, using a fixed seed. Recomputing that partition from the retained game IDs gives:

| Role | Possessions | Games |
|---|---:|---:|
| Model fitting | 896,246 | 4,647 |
| Early-stopping validation | 23,497 | 123 |
| Origin evaluation, 2024–25 | 237,154 | 1,230 |

The saved evaluation table records the following. Its baseline comparison reports ThetaNN NLL **1.114830** and one-hot NLL **2.898532** for the 2024–25 origin. The particularly weak one-hot result and model-vintage dependencies limit interpretation: it is an archived output, not evidence of a well-tuned state-of-the-art comparator. This release does not reload historical checkpoints or independently rerun these historical predictions.

| Historical gate | Saved result | Evidence and limitation |
|---|---|---|
| Points-prediction NLL vs ridge and one-hot | Passed | Paired 90% interval vs calibrated ridge: [-0.001828, -0.001403]; vs one-hot: [-1.815994, -1.753244]; 1,230 games |
| Team-level correlation non-inferiority vs EPM | **Failed** | Difference -0.337188; 90% interval [-0.629129, 0.064923]; 30 teams; does not clear the -0.05 floor |
| Planted player-recovery suite | **Failed** | Separate recorded synthetic suite: trade scenario 1/2 checks, collinear 2/2, opposite-sign 3/3 |
| Within-team permutation comparison | Passed | Recorded real correlation 0.179429 vs permutation mean 0.133475, difference 0.045954 |

The planted suite was not run inside the saved origin-evaluation table; its failed result comes from its separate report. Passing a points-prediction gate or team-permutation check does not rescue the failed player-value checks. **There is no supported overall claim of EPM superiority or of all gates passing.**

EPM is Estimated Plus-Minus, a separate player-value measure from [Dunks & Threes](https://dunksandthrees.com/about/epm). Its underlying player data is not redistributed here. Aggregate gate summaries are provided for methodological transparency, not as a substitute for an independently reproduced comparator study.
