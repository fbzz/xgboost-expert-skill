# XGBoost Optimization Expert

A repository-grade skill for making an LLM behave like a strong **XGBoost performance engineer** instead of a generic hyperparameter suggester.

This project is built around a simple idea: most XGBoost gains do **not** come from random tuning. They come from fixing the evaluation setup, matching the objective to the real business target, identifying overfitting vs underfitting correctly, improving features, and then tuning in a disciplined order.

The included `skill.md` is meant to be dropped into an LLM skill system so the model can diagnose weak XGBoost pipelines, propose better experiments, reduce overfitting, reduce underfitting, and improve out-of-sample generalization with far better structure than ad hoc trial and error.

---

## What this repository is for

Use this repository when you want an LLM to help with:

- tuning XGBoost for tabular problems;
- reducing overfit without killing signal;
- diagnosing underfit instead of blindly increasing complexity;
- improving generalization across folds, time periods, users, entities, or regimes;
- designing a better tuning sequence;
- auditing leakage, split mistakes, metric mismatch, and unstable features;
- producing experiment plans that are small, interpretable, and business-metric aware.

This is especially useful when standard LLM behavior is too shallow, for example:

- suggesting Optuna before checking the split;
- recommending deeper trees because train score improved;
- ignoring calibration when probabilities matter;
- treating feature leakage as a minor detail;
- giving a buffet of knobs instead of a diagnosis.

---

## Core philosophy

The skill is built around five interacting layers:

1. **Problem framing** — objective, target definition, decision metric, leakage boundaries.
2. **Validation design** — split integrity, temporal correctness, grouping, drift, class balance.
3. **Data and features** — signal quality, feature stability, missingness, redundancy, inference safety.
4. **Model capacity and regularization** — depth, child weight, gamma, subsampling, L1/L2, rounds.
5. **Search process and diagnostics** — experiment design, ablations, learning curves, failure analysis.

The model is instructed to treat **generalization** as the main goal.

Not train score.
Not one lucky validation slice.
Not cosmetic parameter movement.

---

## What makes this useful

A good XGBoost assistant should know that the highest-leverage improvements often come from:

- fixing leakage or invalid validation;
- aligning the training objective with the real-world decision objective;
- correcting imbalance handling;
- engineering better, leakage-safe features;
- tuning `learning_rate` jointly with boosting rounds;
- controlling capacity and variance before trying broad search.

This repository encodes that behavior directly into the skill.

Instead of asking the LLM to "be good at XGBoost," it gives the model a strong operating doctrine:

- diagnose first;
- tune in the right order;
- prefer small-batch experiments;
- explain what each result means;
- trust only gains that survive stricter validation.

---

## Who this is for

This repository is meant for:

- ML engineers working with structured/tabular data;
- data scientists tuning production or research XGBoost models;
- teams using LLMs as coding or research agents;
- practitioners who want stronger model debugging, not just hyperparameter search;
- anyone building evaluation-first workflows for classification, regression, ranking, or profit-driven decision systems.

It is particularly helpful in settings where naive tuning wastes time:

- temporal prediction;
- grouped entities such as users, claims, sessions, markets, or devices;
- imbalanced classification;
- noisy labels;
- drifting environments;
- feature-heavy tabular pipelines.

---

## Repository contents

```text
.
├── skill.md        # The main XGBoost skill file for the LLM
└── README.md       # Repository explanation and usage guide
```

If you want, this repository can easily be extended with:

```text
.
├── examples/
│   ├── classification_prompt.md
│   ├── ranking_prompt.md
│   └── temporal_validation_prompt.md
├── templates/
│   ├── overfit_diagnosis.md
│   ├── underfit_diagnosis.md
│   └── experiment_plan.md
└── notes/
    └── xgboost_vs_lightgbm.md
```

---

## What the skill teaches the LLM

The `skill.md` instructs the LLM to reason like a senior XGBoost optimizer.

### 1. Start from the real objective

Before suggesting any tuning, the model must identify:

- the task type;
- the true optimization target;
- the deployment setting;
- the data regime;
- the business metric that actually matters.

That matters because many pipelines are optimized for the wrong thing. For example:

- AUC is not enough if calibrated probabilities matter.
- Accuracy is weak for severe imbalance.
- Classification metrics alone may be irrelevant if the true goal is expected value, recall at K, or trading PnL.

### 2. Check the split before the parameters

The skill forces the LLM to treat split design as a first-class concern.

That means:

- time series should use rolling or expanding validation;
- grouped records should stay together;
- duplicates and related entities should not leak across folds;
- recent regimes should be tested when drift matters.

This is one of the highest-value behaviors in the whole skill.

### 3. Build a conservative baseline first

The skill includes a baseline doctrine built around moderate depth, moderate child weight, some subsampling, and a lower learning rate with early stopping.

The point is to establish a sane reference model before exploring more aggressive configurations.

### 4. Diagnose the failure mode

The LLM is explicitly told to separate:

- **overfitting**;
- **underfitting**;
- **weak signal**;
- **validation weakness**;
- **metric mismatch**;
- **feature instability**;
- **leakage risk**.

This avoids one of the most common problems in model tuning: treating every bad result as if it were just a parameter problem.

### 5. Tune in the right order

The skill teaches a clear order of operations:

1. validation and metric correctness;
2. feature, leakage, and objective fixes;
3. tree capacity;
4. stochasticity;
5. learning dynamics;
6. weight regularization;
7. problem-specific parameters;
8. speed and compute optimization.

That ordering is one of the main reasons this skill is stronger than generic tuning advice.

---

## What kinds of XGBoost problems it handles well

This skill is strongest for repositories or workflows involving:

- binary classification;
- multiclass classification;
- regression;
- ranking;
- tabular feature pipelines;
- time-aware or grouped validation;
- imbalanced tasks;
- decision-oriented model evaluation.

It is also useful when the model appears to be doing well on paper but fails in more realistic validation.

Typical examples:

- validation looks great, but recent data collapses;
- train metrics are strong, validation is unstable;
- the model is too conservative and misses useful interactions;
- probability outputs are poorly calibrated;
- feature importance changes wildly across folds;
- the tuned winner does not beat a simpler baseline under stricter testing.

---

## How to use this repository

### Option 1 — Use the skill directly in an LLM system

Place `skill.md` into your skill or agent framework and invoke it when you want the model to help diagnose or improve an XGBoost setup.

Typical prompt pattern:

```text
Use the XGBoost Optimization Expert skill.

Context:
- Task: binary classification
- Metric: PR-AUC and recall at top 5%
- Data: temporal, grouped by account_id
- Current issue: validation PR-AUC is unstable across folds
- Current params: max_depth=8, min_child_weight=1, subsample=1.0, colsample_bytree=1.0

Please diagnose whether this is overfitting, split weakness, or feature instability, and propose the next 6 experiments in priority order.
```

### Option 2 — Use it as repository guidance for human contributors

Even without an automated skill framework, the file works as a strong internal standard for how the team should approach XGBoost tuning.

It can act as:

- a modeling doctrine;
- a review checklist;
- a prompt source for code agents;
- a training document for new contributors.

---

## Expected behavior from the LLM

When used correctly, the LLM should stop acting like a random parameter assistant and start producing responses like:

- a diagnosis of the likely failure mode;
- the reasoning behind that diagnosis;
- a prioritized sequence of small experiments;
- an explanation of what each experiment would prove or disprove;
- a concrete recommendation for the best next move.

The skill also includes response templates such as:

- overfit fix;
- underfit fix;
- generalization-first search.

These make the outputs more consistent and more useful in real project workflows.

---

## Why this repository is not just "hyperparameter tuning"

The repository is intentionally broader than parameter search.

That is because XGBoost quality depends on more than the knobs.

The biggest real-world wins often come from:

- correcting label leakage;
- redesigning temporal validation;
- removing fragile feature families;
- using the right metric for the actual decision problem;
- understanding whether the model is signal-limited;
- comparing against simpler baselines under harder evaluation.

This repository encodes those behaviors so the LLM does not over-focus on `max_depth`, `gamma`, or `reg_alpha` at the wrong time.

---

## Overfitting, underfitting, and generalization

A major purpose of the skill is to help the LLM reason about three common failure modes.

### Overfitting

The skill pushes the model to look for signs such as:

- train score far above validation;
- high fold variance;
- unstable feature importance;
- overconfident probabilities;
- gains that disappear under stricter validation.

And then respond with the correct playbook:

- reduce depth;
- increase child support;
- add pruning pressure;
- add row and column sampling;
- lower learning rate and rely on early stopping;
- simplify unstable features;
- verify the split again.

### Underfitting

The skill also covers the opposite case, where the model is too rigid.

The LLM is guided to notice when:

- train and validation are both weak;
- early stopping happens too early;
- useful interactions may be missing;
- the current configuration is overly conservative.

Then it can recommend:

- moderate increases in tree depth;
- lower child weight;
- less pruning pressure;
- more rounds;
- less randomness if the model is too weak;
- better features when the issue is not parameter capacity.

### Poor generalization

The most important concept in the repository is that apparent gains are not real unless they hold up under proper validation.

This is why the skill emphasizes:

- fold stability;
- temporal robustness;
- feature-family ablations;
- simpler baseline comparisons;
- sensitivity checks around thresholds and calibration.

---

## Design principles behind the skill

This repository follows several design principles.

### Evaluation-first

No tuning advice should come before split validation.

### Small-batch experiments

The skill prefers interpretable experiments over giant unguided search.

### Root-cause reasoning

The LLM should explain *why* a model is weak, not just what to try next.

### Business-metric alignment

The model should optimize what actually matters downstream.

### Generalization over cosmetics

A smaller but robust improvement is worth more than a flashy but fragile one.

---

## Anti-patterns this repository tries to prevent

This repository is explicitly designed to reduce common modeling mistakes, including:

- tuning on an invalid random split for temporal data;
- trusting train score improvements;
- reporting one lucky validation result;
- using accuracy on imbalanced tasks;
- adding target encoding without leakage-safe construction;
- blindly setting `scale_pos_weight` and trusting raw probabilities;
- launching broad HPO before understanding the failure mode;
- confusing faster GPU training with better model quality;
- celebrating tiny metric gains that sit inside fold noise.

---

## Recommended workflow in a real project

A practical way to use this repository in a modeling repo is:

1. define the task, metric, and deployment context;
2. document the split strategy;
3. establish a conservative baseline;
4. run a small diagnostic experiment set;
5. classify the main failure mode;
6. tune capacity and randomness in the correct order;
7. improve features if the model is still signal-limited;
8. run bounded HPO only after the system is understood;
9. confirm the winner on stricter validation;
10. document why the final model is trusted.

That sequence is close to how strong human practitioners already work, and the skill is meant to push the LLM into that same behavior.

---

## Suggested next additions

If you want to make this repository stronger, the best follow-up additions would be:

- example prompts for classification, ranking, and temporal modeling;
- a checklist for leakage audits;
- an experiment-tracking template;
- a model review template for PRs;
- side-by-side comparison guidance for XGBoost vs LightGBM;
- sample diagnostics on synthetic overfit and underfit cases.

---

## Summary

This repository gives you an LLM skill that treats XGBoost tuning as a **generalization problem**, not a knob-turning problem.

It is designed to produce better diagnoses, cleaner experiment plans, stronger overfit and underfit handling, and more trustworthy out-of-sample improvements.

If your goal is to make an LLM genuinely useful for XGBoost optimization work, this is the right foundation.

