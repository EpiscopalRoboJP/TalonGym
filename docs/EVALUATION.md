# Evaluation and comparison

Do not pick a strategy from one lucky rollout. TalonGym ranks policies on **true AUTO score** over held-out seeds, with bootstrap confidence intervals.

## CLI (scripted baseline)

```bash
python -m talongym evaluate --trials 32
python -m talongym evaluate --checkpoint var/ckpts/best.zip --trials 32
```

Default runs [`scripted_auto`](../python/talongym/training/policies.py) on the active bundle. Lab Compare can eval a **trained** checkpoint from a run. Seeds are `10_000_000 … 10_000_000 + n - 1` (same start as training-preset `heldOutSeedStart`).

Printed fields:

- `mean` — mean true score
- `95% CI [lo, hi]` — bootstrap on the mean (2000 resamples)
- `p10` — 10th percentile
- `n` — trial count
- `eligible` — `true` only if n ≥ 500

32 trials is a sanity check. A label-eligible report is n = 500 (preset default). Cap in the Lab API is 500.

Scripted AUTO and checkpoint evals fan out across CPU processes (`evalWorkers` defaults to at most 8). Lambdas and other unpicklable policies stay sequential. Override with `TALONGYM_EVAL_WORKERS=1` to debug. Checkpoint copies in workers load on CPU so they do not clone the GPU policy N times.

## Lab Compare

Open `/compare` ([LAB.md](LAB.md)).

1. **Run 24-trial evaluation (scripted)** — stores a row; UI keeps it unlabeled (`n<500`).
2. Pick a finished training run and **Optional checkpoint eval** — loads that run’s checkpoint artifact and the run’s field/robot/scoring ids.
3. Read mean, CI whiskers, p10, collision time, restricted-entry rate.
4. Export Road Runner from the eval’s best-scoring replay, then paste into an AUTO OpMode (Control Hub runs it; this is the only field path).

The page refuses a “best” label when any of:

- any row has n &lt; 500
- fewer than one evaluation
- the top two 95% CIs overlap

That matches `GET /api/v1/comparisons/latest` → `bestLabelEligible`.

## API

```http
POST /api/v1/evaluations
{
  "nTrials": 500,
  "objective": "mean_true_score",
  "policy": "scripted"
}
```

`policy`: `scripted` (default) or `checkpoint` / `trained` with `runId` and/or `checkpoint` path. n is clamped to `[8, 500]`. The request is **synchronous** (HTTP 200). `objective` (`mean_true_score` \| `p10_true_score` \| `lcb_true_score`) is stored on the report as `objective` / `objectiveValue`. The best trial’s frames are stored as a replay (`report.replayId`).

## Report fields

From [`python/talongym/eval/harness.py`](../python/talongym/eval/harness.py):

| Key | Meaning |
|-----|---------|
| `mean`, `lo`, `hi`, `median`, `p10`, `p90`, `min`, `max` | True-score distribution + bootstrap CI on the mean |
| `nTrials`, `seeds`, `scores` | Raw trials |
| `collisionRate`, `collisionTimeMean`, `firstContactS` | Contact with walls/robots |
| `restrictedEntryRate` | Data rule / accumulator `restricted_entry` (G402-style), not a hardcoded season name. The same node also deducts −15 from `trueScore` (training foul proxy); the rate remains a side metric. |
| `bestScore`, `bestFrames` | Highest true score this batch (frames dropped after the API saves a replay) |
| `objective`, `objectiveValue` | Requested objective and the scalar used for it (mean / p10 / LCB) |
| `evalWorkers` | Process count used for this batch (`1` when sequential) |
| `bestLabelEligible` | n ≥ 500 for a single batch |

Shaping never appears in this report.

## Paired comparisons

`run_paired_trials` (tests / CRN) can evaluate two policies on the **same seeds** (`pairCommonRandomNumbers`). Overlapping CIs ⇒ `bestLabelEligible: false`. Use this when ranking two trained checkpoints; Holm correction for more than two policies is specified in [ARCHITECTURE.md](ARCHITECTURE.md) but the Lab latest-comparison is a pairwise overlap check on the top two rows.

## Objectives

Training may optimize `mean_true_score`, `p10_true_score`, or `lcb_true_score`. The CLI evaluate printout is always mean + CI + p10. AUTO-only RP-proxy numbers in scoring JSON are **not** “Ranking Points earned” — do not label them that way in notes or UI copy.
