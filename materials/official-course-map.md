# Official course map - learn-ml-epistemology-with-phoebe

Built 2026-09-19. Hub bucket `ds` (Data Science), difficulty tier 4. Flips the hub's existing
`planned` entry live. Two tracks: leader 6 x 45 min, practitioner 10 x 45 min.

Running artifact: **a belief you can defend** about one model - the deterioration-risk triage
model at **Thornbury**, a fictional hospital group. Every true relationship is written into the
generator, so what the model learned and what it should have learned can both be checked.

---

## The seam - read this before writing a single page

**This course was re-scoped on 2026-09-19.** The hub's blurb promised memorization vs
generalization, double descent, a leakage taxonomy, benchmark illusions, Goodhart and
underspecification. **Three of those six are already taught in depth elsewhere.** What no course
in the estate owns is the question underneath all of them: **what licenses the belief that a model
will work on data it has not seen.** `D'Amour`, `Kapoor`, `test set reuse`, `adaptive overfitting`,
`benchmark lottery` and `underspecification` all returned **zero hits estate-wide** before this
build.

| Sibling | What it already owns, live | What this course does instead |
|---|---|---|
| `learn-deep-learning-with-phoebe` (ds, d4) | Session 05 "Taming the net": the Zhang et al. random-labels result with its exact scope, the interpolation threshold, label noise, and "one honest sentence about double descent" | **Memorization and double descent are not re-taught.** Named once in a clause that hands them to session 05. The question here is not how a net can fit noise, it is what its test score entitles you to claim |
| `learn-model-risk-with-phoebe` (ds, d3) | p3 "The leak hunt": target leakage as a validation finding - timing, how a post-outcome field reaches a feature list, why a noisy leak survives review, filing it as conceptual soundness | **Leak mechanics are not re-taught.** This course takes leakage at the level of a *field*: Kapoor and Narayanan's survey of 294 papers across 17 disciplines, and what it means that a whole literature was wrong in the same way |
| `learn-okr-with-phoebe` (lead), `learn-model-evaluation-with-phoebe` (ds), `learn-ai-evals-with-phoebe` (ai) | Goodhart as metric design: b9 "When metrics go wrong", a5 "Three project killers", a4/a5 scorecard and drift | **Metric gaming is not re-taught.** Goodhart appears only as an epistemic case: what happens to a *measurement* when it becomes a target, and why a benchmark is a measurement |
| `learn-model-evaluation-with-phoebe` (ds, d3) | Every metric and how to compute it; drift detection; the eval scorecard | **No metric is defined here.** Accuracy appears only as a claim whose warrant is in question |
| `learn-causal-inference-with-phoebe` (ds, d4) | Identification, what not to adjust for, the causal model | Untouched. Where a page would drift into causal language it names the course and stops |
| `learn-experimentation-with-phoebe` (ds, d3) | External validity for experiments, geo and synthetic control | Named once. Generalisation of an *experiment* is theirs; generalisation of a *fitted model* is here |

**What this course uniquely owns:** the inductive assumption a held-out set rests on and when it
breaks; underspecification and the stress test; what selecting on a test set does to the number you
report, measured, against the field evidence that says it matters less than feared; benchmark
construct validity and the benchmark lottery; leakage as a reproducibility crisis across
disciplines; and the question that ends every review - what evidence would change your mind.

---

## Verified facts (with their source tier)

**Tier 1, primary papers, fetched and checked on 2026-09-19. Every quotation below is verbatim
from the source, not from a summary.**

- **D'Amour, Heller, Moldovan, Adlam, Alipanahi, Beutel, Chen, Deaton, Eisenstein, Hoffman,
  Hormozdiari, Houlsby, Hou, Jerfel, Karthikesalingam, Lucic, Ma, McLean, Mincu, Mitani,
  Montanari, Nado, Natarajan, Nielson, Osborne, Raman, Ramasamy, Sayres, Schrouff, Seneviratne,
  Sequeira, Suresh, Veitch, Vladymyrov, Wang, Webster, Yadlowsky, Yun, Zhai and Sculley (2020),
  "Underspecification Presents Challenges for Credibility in Modern Machine Learning"**,
  arXiv 2011.03395. The definition, verbatim: **"An ML pipeline is underspecified when it can
  return many predictors with equivalently strong held-out performance in the training domain."**
  And: "Predictors returned by underspecified pipelines are often treated as equivalent based on
  their training domain performance, but we show here that such predictors can behave very
  differently in deployment domains." Their stress tests span **computer vision, medical imaging,
  natural language processing, clinical risk prediction from electronic health records, and
  medical genomics**. Forty authors; cite as "D'Amour et al." and never invent a short author list.
- **Recht, Roelofs, Schmidt and Shankar (2019), "Do ImageNet Classifiers Generalize to
  ImageNet?"** They built new test sets and measured **accuracy drops of 3 to 15 percent on
  CIFAR-10 and 11 to 14 percent on ImageNet**. **The finding that matters most for this course is
  their conclusion about the cause:** the drops came from the models' limited ability to
  generalise to slightly harder images, **not** from test-set contamination through repeated use.
  They also note "accuracy gains on the original test sets translate to larger gains on the new
  test sets".
- **Roelofs, Shankar et al. (2019), "A Meta-Analysis of Overfitting in Machine Learning",
  NeurIPS 2019.** Over **one hundred Kaggle competitions**, where entrants repeatedly score against
  a public holdout and a separate test set decides the final ranking. They found **little evidence
  of substantial adaptive overfitting** despite the repeated evaluation.
- **Kapoor and Narayanan (2023), "Leakage and the reproducibility crisis in machine-learning-based
  science", Patterns.** Leakage found in **17 fields**, collectively affecting **294 papers**, with
  a taxonomy of **eight types** ranging from textbook errors to open research problems.
- **Dehghani, Tay, Gritsenko, Zhao, Houlsby, Diaz, Metzler and Vinyals (2021), "The Benchmark
  Lottery"**, arXiv 2107.07002. Verbatim: the benchmark lottery "postulates that many factors,
  other than fundamental algorithmic superiority, may lead to a method being perceived as
  superior", and they show "the relative performance of algorithms may be altered significantly
  simply by choosing different benchmark tasks".

**The live disagreement this course must teach rather than settle.** Theory says repeatedly
selecting against a held-out set inflates the number it reports. Two strong empirical results say
the effect is mild in real practice: Recht et al. attribute the ImageNet and CIFAR-10 drops to a
distribution gap rather than reuse, and Roelofs et al. find little adaptive overfitting across a
hundred competitions. **Bench B measures the mechanism and finds it real but small**, which is
consistent with both. No page may claim that leaderboards are broadly corrupted by reuse; no page
may claim reuse is harmless either. State the mechanism, state the measurement, state the field
evidence, and let the reader hold all three.

**Tier 2, named for orientation, never taught as a recipe.**

- Dwork et al.'s reusable holdout and differential privacy as a proposed fix for adaptive
  analysis: named once, not derived.
- Zhang et al.'s random-labels result and double descent: named once and handed to
  `learn-deep-learning-with-phoebe` session 05.

---

## Frozen canon - the Thornbury benches

Computed in node from `assets/ep-live.js` before any page quoted a number. Every figure is a real
network trained in the browser by stochastic gradient descent and evaluated by counting, not a
stated result. Any page citing these must match exactly.

**The world.** Thornbury predicts whether a patient deteriorates within 24 hours. Two honest
signals (a vitals drift and a lab trend) genuinely cause the outcome. One **shortcut** does not:
which ward the patient is on. In the collection year the high-acuity ward received most of the
deteriorating patients, so ward agrees with the outcome **90 percent** of the time without causing
it. In the deployment year the wards are reorganised and ward agrees **50 percent** of the time:
the shortcut is dead. Training 4,000 patients, held-out 2,000 from the collection year, shifted
2,000 from the deployment year.

### Bench A - underspecification, measured

Twenty models per data draw. **The only thing that differs between them is the random seed**,
which sets the initial weights and the order the examples arrive in. Same data, same architecture
(3 inputs, 12 tanh hidden units, 1 logit), same schedule.

| Data draw | Held-out spread | Shifted spread | Ratio | corr(held, shifted) |
|---|---|---|---|---|
| 1 | 0.019 | 0.051 | 2.7x | -0.89 |
| 2 | 0.008 | 0.038 | 4.8x | +0.03 |
| 3 | 0.012 | 0.046 | 4.0x | -0.87 |
| 4 | 0.003 | 0.034 | 11.3x | +0.09 |
| 5 | 0.016 | 0.048 | 3.0x | -0.75 |

**Mean held-out accuracy 0.949. Mean shifted accuracy 0.765.** Behavioural reliance on the
shortcut, measured by flipping the ward on every patient and counting how many predictions change,
runs from **0.70 to 1.00** across seeds: some models change their mind about 7 patients in 10 when
only the ward changes, others about all of them.

**What is robust, in five draws out of five:** the shifted spread is larger than the held-out
spread, by **2.7 to 11.3 times**. Models that are indistinguishable where you looked are
distinguishable where you did not.

**What is NOT robust, and the course says so:** the correlation between held-out and shifted
accuracy swings from **-0.89 to +0.09** depending on the data draw. **A first run of this bench
produced -0.89 and the finding "the best model on the held-out set is the worst under shift". One
more data draw destroyed it.** Held-out accuracy does not predict shifted accuracy, and it does not
reliably anti-predict it either; it is close to uninformative about it. That near-miss is itself
course material and is taught in p9, because it is exactly the failure the course is about.

### Bench B - what selecting on a test set costs

A pool of k candidate models, all scored on **one** test set. The analyst keeps the best and
reports its test accuracy. The honest number is that same model's accuracy on 6,000 fresh patients
nobody selected on. Candidates are smaller nets (6 hidden units, 8 epochs); test set 400 patients;
each row the mean of 12 replications.

| Candidates k | Reported | Honest | Optimism |
|---|---|---|---|
| 1 | 0.9527 | 0.9500 | 0.0027 |
| 2 | 0.9544 | 0.9499 | 0.0045 |
| 5 | 0.9548 | 0.9499 | 0.0049 |
| 10 | 0.9552 | 0.9499 | 0.0053 |
| 25 | 0.9562 | 0.9494 | 0.0068 |
| 50 | 0.9567 | 0.9495 | 0.0072 |
| 100 | 0.9569 | 0.9497 | 0.0072 |

**The honest column does not move.** Selecting harder bought **0.0042 of reported accuracy and
-0.0003 of real accuracy** between k=1 and k=100. The optimism grows with k and then saturates,
and it never reaches one accuracy point.

**Test-set size is the lever that matters** (k = 60, mean of 40 replications, with the standard
error of the mean):

| Test n | Optimism | SE |
|---|---|---|
| 100 | 0.0071 | 0.0030 |
| 200 | 0.0091 | 0.0022 |
| 400 | 0.0062 | 0.0017 |
| 1000 | 0.0044 | 0.0010 |
| 2000 | 0.0030 | 0.0006 |

From 200 upward the trend is clean and well outside the error bars: **a bigger test set buys back
the optimism**. At n = 100 the estimate is 0.0071 with a standard error of 0.0030, and it cannot be
ranked against n = 200 - **at very small test sets even the measurement of the bias is too noisy to
trust**, which is its own lesson and must be stated wherever that row appears.

**Known canon gap (p6).** The tables above give the standard error of the *mean optimism* over
replications. They do not give the standard error of a *single accuracy estimate* at each test-set
size. p6 therefore quotes no numeric interval width and computes the intervals live from the
engine instead. If that table is ever computed, add it here first, then quote it beside the SE
column in p6 Part 1 and add a cheat-sheet line for it.

**Known canon gap (p8).** The map records the count of eight leakage types and the range they
span (textbook errors to open research problems) but not the individual type names, so no type is
named on p8. Verify the eight names against Kapoor and Narayanan at source and add them here
before any page names them.

**A claim NOT to make.** Do not write that leaderboards are broadly corrupted by test-set reuse, or
that "everyone is overfitting the benchmark". The measured optimism here is under one point and
saturates, Recht et al. attributed the ImageNet drops to a distribution gap rather than reuse, and
Roelofs et al. found little adaptive overfitting across a hundred competitions. The defensible
claim is narrower and more useful: **selecting on a set converts it from evidence into a target,
the cost is measurable and modest, and it scales with how small the set is.**

---

## Coverage per session

`✓` = taught to working depth. `◐` = named and handed to the session or course that owns it.

### Leader track

| Session | Covers | Depth |
|---|---|---|
| a1 What a test score entitles you to | The inductive assumption behind a held-out number; the three things it silently assumes; 0.949 against 0.765 as the price | ✓ |
| a2 Equally good, differently wrong | Underspecification in plain terms; twenty models, one pipeline, a shifted spread 2.7 to 11.3 times the held-out spread | ✓ |
| a3 When the number is a target | Goodhart as an epistemic problem; what selecting on a set does to it; the measured optimism and the field evidence against alarm | ✓ |
| a4 What a benchmark licenses | Construct validity; the benchmark lottery; why a leaderboard rank is not a claim about your problem | ✓ |
| a5 A whole literature, wrong the same way | Kapoor and Narayanan: 294 papers, 17 fields, 8 leakage types; what it means for evidence you did not produce | ✓ |
| a6 What would change your mind | The question to ask before approving; writing down the falsifier in advance | ✓ |
| Double descent, memorization | Handed to `learn-deep-learning` session 05 | ◐ |
| Leak mechanics in one model | Handed to `learn-model-risk` p3 | ◐ |

### Practitioner track

| Session | Covers | Depth |
|---|---|---|
| p1 The assumption under the split | What i.i.d. buys and what it does not; when a random split is a lie about the world | ✓ |
| p2 The shortcut | Building the Thornbury world; the ward that predicts without causing; measuring behavioural reliance by flipping it | ✓ |
| p3 Twenty seeds, one pipeline | Bench A: identical held-out, divergent shifted; the ratio and what it means | ✓ |
| p4 Stress tests | Designing the shifted set; what a stress test can and cannot establish; D'Amour's five domains | ✓ |
| p5 Selecting on a set | Bench B: the k sweep, the flat honest column, the saturation | ✓ |
| p6 How big does a test set need to be | The size sweep with error bars; why n = 100 cannot even measure its own bias | ✓ |
| p7 Reading a benchmark | Construct validity, the benchmark lottery, what transfers to your problem | ✓ |
| p8 Leakage at the scale of a field | The eight-type taxonomy; reproducing a leak; what a reviewer could have caught | ✓ |
| p9 The finding that did not survive | The -0.89 correlation that one more seed destroyed; single-draw findings; how to test your own claim before publishing it | ✓ |
| p10 The credibility bench | Both panes live, plus the falsifier written before the run | ✓ |
| Metric definitions, drift detection | Pointed at `learn-model-evaluation` | ◐ |

## Not covered, by design

- **Memorization, random labels, double descent.** `learn-deep-learning` session 05.
- **Leak mechanics inside one validation.** `learn-model-risk` p3.
- **Metric design and gaming.** `learn-okr`, `learn-model-evaluation`, `learn-ai-evals`.
- **Causal identification.** `learn-causal-inference`.
- **Differential privacy and the reusable holdout in technical detail.** Named once in p5.
- **Fairness auditing.** Stress tests here are about credibility, not about group harm; the
  fairness literature is not taught.

## Re-verify before delivery

The five papers are settled publications and will not move. The benches are deterministic from
fixed seeds: if `ep-live.js` is edited, re-run the node harness and update every number in this
file before touching a page. **Any new bench claim must be run across at least five data draws
before it is written down** - the -0.89 correlation looked like the headline of the course until
the fourth draw returned +0.09.
