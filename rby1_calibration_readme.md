# Calibrating an RB-Y1 against motion capture

End-to-end guide: rosbag → dataset → `AX = YB` calibration → results you can defend.

This is the workflow that produced the Aug 25 2026 results (`Y = base_T_map` at
`(1447.43, 64.51, −10.45) mm`, ~1.7 mm working accuracy). It assumes the robot
carries mocap marker clusters on several links and that you record both `/tf` and
`/rigid_bodies` while the robot visits a set of stationary poses.

---

## 1. What is being solved

```
A X = Y B

A = base_T_link     robot forward kinematics, from /tf
B = map_T_rigid     motion capture, from /rigid_bodies
X = link_T_rigid    where the marker cluster sits on the link   (unknown, one per target)
Y = base_T_map      where the mocap world sits relative to the robot   (unknown, shared)
```

`Y` is the same physical transform for every target, which is what makes
cross-target agreement a real accuracy check rather than a self-report.

---

## 2. Collect the data

The robot visits stationary poses; at each one you hold still and record. The
builder expects these topics (all overridable):

| purpose | default topic |
|---|---|
| window start / end | `/goto_random_configs_n/window_start`, `.../window_end` |
| epoch / datapoint index | `.../epoch_number`, `.../datapoint_number` |
| joint states | `/joint_states` |
| kinematics | `/tf`, `/tf_static` |
| motion capture | `/rigid_bodies` |

### Collection guidance — read before recording

These are the things that went wrong on the Aug 25 capture and cost two targets.

**Rotate about at least two non-parallel axes, per target.** `AX = YB` separates
`X` from `Y` only then. If every rotation shares one axis, you can turn `X` about
it and counter-turn `Y` by the same amount and the data fits exactly as well — an
entire family of solutions, none distinguishable. The failure is silent: you get a
healthy residual next to a wildly wrong `Y`.

**Move the torso.** On the Aug 25 capture it never moved, which cost two targets at
once. The torso itself became unsolvable, and so did `link_right_arm_0` — as the
first link of the arm its pose changes only when joint 0 turns, about one fixed
axis, so no amount of *arm* motion can give it a second one. Torso rotations
compose with joint 0's and fix both.

**Vary the wrist.** Distal links need orientation diversity, not just position
diversity, to be well conditioned.

**Repeat each configuration 3 times, and spend the rest of your budget on more
configurations.** Repeats are worth having, but not for the reason they look like —
see "How many repeats" below.

**Keep every marker visible.** On Aug 25, 36 of 700 observations were dropped for
invalid markers and 1 more for having too few repeats left afterwards. Check
occlusion before a long capture, not after.

### How many repeats

Repeats measure something real, and it is worth being precise about what.

**What the repeat scatter is not: sensor noise.** Each window averages ~1200 mocap
frames, which drives the noise on a window *mean* down to about 0.0016 mm. The
observed repeat-to-repeat scatter is 0.12–0.28 mm depending on the link — roughly
**100× that floor**. It is real physical non-repeatability: the robot does not return
to the same pose. Forward kinematics cannot see it (its own repeat scatter is
0.007–0.046 mm, purely encoder repeatability, because the encoders return to the same
counts whether or not the link does), so motion capture is the only instrument here
that measures it. Session drift does not explain it either — between-epoch
differences account for only 5–6% of the variance.

**What repeats cannot give you: a per-configuration covariance.** A covariance from
`n` samples has rank `n − 1`, so a 6-DoF pose covariance needs `n ≥ 7` merely to be
full rank, and the relative error on a variance estimate is `sqrt(2/(n−1))` — 71% at
`n = 5`, 100% at `n = 3`. Any single configuration's estimate is very noisy. What
makes it usable is fitting sigma(q) as a *smooth* function, so neighbouring
configurations share information — which is why coverage matters more than depth.

**Non-repeatability varies a lot between configurations** — 0.019 mm to 0.838 mm on
the Aug 25 data, a 40× range. Whether that variation is *predictable* from
configuration is untested: leave-one-configuration-out skill ran from −0.83 to +0.20,
which at 20 configurations means "unknown", not "no". Settling this is one of the
things a larger capture buys.

**Outlier detection.** 2–3 configurations per target run 4–8× above the rest, mostly
body-specific rather than pose-global (only configuration 8 was flagged by 4 of 7
targets, the rest by one or two), which points at individual rigid bodies losing
markers. Three repeats let a majority vote identify the odd one; two only tell you
that two observations disagree.

**Path dependence — tested, and not present in this data.** The stronger argument for
many repeats is that the error at a pose may depend on how the arm got there
(backlash, hysteresis), so repeats sample the distribution over approach paths. The
Aug 25 capture tests this by construction: the epochs are shuffled, so each
configuration was reached from ~4.7 distinct predecessors across its 5 repeats. Two
tests find nothing:

- *Directional hysteresis.* The repeat-to-repeat error component along the direction
  of travel into the pose averages 0.000–0.009 mm, all `|t| < 1.5`, and the spread is
  roughly isotropic (along/perpendicular 0.4–1.3×) rather than concentrated along the
  travel direction.
- *Predictability.* A regression from the predecessor's pose to the
  configuration-centred error scores skill `-0.03` to `0.00` under
  leave-one-configuration-out. The predecessor carries no information.

That is physically plausible — the RB-Y1 uses harmonic drives, which have very little
backlash. Caveat on power: the test would not see an effect below ~0.05 mm, and it
uses the Cartesian approach direction as a proxy because the dataset carries no joint
states. Testing per-joint approach *sign*, which is what backlash actually depends on,
needs the joint-state export.

**Spend the budget on configurations, at 3 repeats each.** Repeats and configurations
come out of the same total, so the question is always what a marginal visit buys. Both
quantities you want to predict — systematic error (0.38–1.06 mm) and non-repeatability
(0.12–0.28 mm) — are functions of configuration over a 7-DoF space, and extra repeats
at a point you already sampled teach you nothing about the shape of either.

| total visits | good split | configurations | noise dof |
|---|---|---|---|
| 1200 | 3 × 400 | 400 | 800 |
| 1500 | 3 × 500 | 500 | 1000 |
| 2000 | 4 × 500 | 500 | 1500 |

Prefer more configurations over more repeats until you reach 3; past 4 the return is
poor. Sample the joint space with a space-filling design (Latin hypercube or a
low-discrepancy sequence) rather than uniform random draws — at a few hundred points
in 7 dimensions, coverage is the binding constraint and random sampling wastes it on
clusters.

**If path dependence is still a live hypothesis, test it separately.** A dedicated
experiment is the only thing with the power to settle it: with sigma ≈ 0.165 mm, a
design needs ~170 samples per approach direction to resolve a 0.05 mm effect, so about
1400 observations for 8 directions. Note the size before committing — a plausible
harmonic-drive backlash effect is 10–40× smaller than the systematic error you are
trying to model, so it is usually not worth the budget.

---

## 3. Build the dataset

From the data root (the directory holding `aug_25_2026/`):

```bash
./aug_25_2026/run_dataset_pipeline.sh rosbag2_2026_08_26-05_07_33_0.db3
```

`BAG` may be a `.db3` file, a rosbag directory containing exactly one, or a name
under `aug_25_2026/`. An optional second argument sets the output root. Set
`PYTHON_EXECUTABLE` if your ROS packages live behind a different interpreter.

The pipeline runs five stages: extract stationary windows → cross-set outlier
summaries → source noise plots → apply marker/repeat filtering → filtered plots.
It writes two directories:

```
probabilistic_axyb_dataset/                 all windows, provenance
probabilistic_axyb_dataset_valid_markers/   filtered — use this one
```

Filtering keeps a target/window only when every mocap frame has at least one finite
marker, then drops any link/`unique_id` left with fewer than two repeats.

To pass builder options through:

```bash
./aug_25_2026/run_dataset_pipeline.sh BAG OUTPUT_ROOT -- \
  --base-frame base --exclude-target 11 --target-map RIGID=TF
```

Individual stages, if you need them:

```bash
python3 build_probabilistic_axyb_dataset.py --bag BAG --output probabilistic_axyb_dataset
python3 filter_valid_markers_dataset.py --source probabilistic_axyb_dataset \
                                        --output probabilistic_axyb_dataset_valid_markers
python3 visualize_noise.py --dataset probabilistic_axyb_dataset_valid_markers
```

### What the calibration actually reads

Only two keys per `<target>.npz`:

```
A   (n, 4, 4)   base_T_link
B   (n, 4, 4)   map_T_rigid
```

Everything else — `unique_id`, covariances, `inv_Sigma_*`, marker counts — is
ignored by the solver. **Any dataset with those two arrays works**, so you are not
tied to this bag format.

---

## 4. Run the calibration

```bash
cd /path/to/probabilisticAXYB
python -m examples.calibrate_rby1 /path/to/probabilistic_axyb_dataset_valid_markers
```

About 87 s for 7 targets × ~100 poses. Options:

| flag | effect |
|---|---|
| `--iterations N` | noise re-estimation passes (default 5; converges by 3) |
| `--skip-sensitivity` | drop the mocap-as-noiseless cross-check |

Keep a record: `... > calib_$(date +%F).txt`

Verify the library first if you have just pulled:

```bash
python -m unittest discover -s tests
```

---

## 5. Set the noise model

You set exactly one thing — the mocap anchor, at `examples/calibrate_rby1.py:48-49`:

```python
MOCAP_SIGMA_POSITION = 0.3e-3           # metres,  from your spec sheet
MOCAP_SIGMA_ROTATION = np.deg2rad(0.5)  # radians, from your spec sheet
```

**Why an anchor is mandatory.** Only the *ratio* of the FK and mocap covariances is
identifiable. Fix neither and the problem has no unique answer — the solver trades
FK error against mocap error along a direction the data cannot see. Mocap is the
side with a trustworthy number, so it is the anchor. Forward-kinematics noise is
then whatever is left in the residual, re-estimated each pass and pooled across all
of a target's poses.

`FK_SIGMA_*` (lines 54-55) are iteration *seeds*, overwritten after the first pass.
Order of magnitude is all that matters; do not tune them.

### Three things not to do

**Don't use the `inv_Sigma_*` arrays from the NPZ.** Estimated from 2–5 repeats
each, they are rank deficient by construction (two samples give a rank-one 3×3).
Inverting them, even with an eigenvalue floor, yields near-hard constraints along
essentially arbitrary directions. They fit measurably worse than plain isotropic
weighting. The script ignores them deliberately.

**Don't derive FK noise from repeat variance.** Repeat spread measures *precision,
not accuracy*. Return to the same commanded joint configuration and FK spread reads
~6 µm — that says the encoders repeat, not that the kinematics are right. Link
lengths, joint offsets and gravity deflection are identical on every repeat and
therefore invisible. The same caveat applies to the mocap spec, which is accuracy
including volume-dependent bias.

**Don't lower the mocap anchor to "resolve FK better."** It doubles as the floor;
below it you are reading estimator artefact, not measurement.

### If mocap is noiseless

Configuration 3 treats **`A`** as exact, which is backwards for this system — FK is
the noisy side here. To put mocap in the noiseless slot instead, no new solver is
needed: `AX = YB` implies `B X⁻¹ = Y⁻¹ A`, so feed the swapped pair to
`solve_axyb_prob_noiseless_a(B, A, inv(X0), inv(Y0), ...)` and invert the results.
The script runs this as a sensitivity check.

---

## 6. Read the output

Six sections, in the order they should change your mind.

**Excitation screening — a gate.** Below `1e-3` a target cannot determine `Y` and is
deferred to stage 2. If a target you expected to move appears there, the problem is
the data, not the solver.

**Measurement consistency — read this before the calibration.** For any pose pair,
`Aᵢ⁻¹Aⱼ` and `Bᵢ⁻¹Bⱼ` describe the same motion in two frames, so their rotation
angles and screw-axis displacements must agree. Neither `X` nor `Y` appears, so no
modelling choice can flatter it. The `versus mocap spec` multiplier is the
diagnosis: ≈1× means that channel is mocap-limited and cannot be improved without
better mocap; ≫1× means something else dominates. Aug 25 read 0.5–1.6× in rotation
and 2.2–7.8× in translation — rotation mocap-limited, translation FK-limited.

**Calibration block.** Check in order: (1) `converged` on every line — discard any
target without it; (2) FK σ stable by pass 3–4; (3) `(at mocap floor)` means the
residual cannot resolve FK error in that channel because it sits below mocap
resolution. That is a statement about observability, not a failure.

**Cross-target `Y` agreement — your error bar.** `Y` is physically one transform, so
disagreement between targets is system error with no assumptions in it. Compare it
against the reported σ. Aug 25: 5.75 mm spread against 1.34 mm reported, 4.3×.
**Quote the spread.** The covariance formula assumes zero-mean independent noise;
systematic FK bias is neither.

**Stage 2.** `X` for deferred targets, computed directly as `X = A⁻¹ Y B` once `Y`
is pinned — not a second optimisation. Its scatter is a consistency figure, not
accuracy: error in `Y` transfers into `X` at close to 1:1, so carry the `Y` spread.

**Sensitivity.** Shifts under 1× σ mean the answer is model-independent.

### Which number to quote

`Y`'s translation is the position of the mocap origin, ~1.45 m from the base — the
worst place to read the error, because a 0.2° angular disagreement becomes
millimetres at that distance. Aug 25:

| evaluated at | disagreement |
|---|---|
| robot workspace (real EE positions) | **1.66 mm rms, 3.22 mm max** |
| mocap origin (what `Y`'s translation reports) | 5.75 mm |

Quote the workspace figure.

---

## 7. Troubleshooting

| symptom | cause | fix |
|---|---|---|
| `transform motions are degenerate; rotations share a single axis` | single-axis or stationary target | expected for such targets; the example script defers them to stage 2 |
| `NOT CONVERGED` on a target | bad initialisation or pathological weighting | discard that target; check its excitation and residual |
| residual ~100° / ~300 mm | pre-fix `solve_axyb` sign bug | ensure `initialization.py` has the `det(vx) < 0` sign flip |
| `Y` metres from plausible | degenerate target that slipped the guard | check excitation; a low residual does not imply identifiability |
| FK σ pinned at the anchor | FK error below mocap resolution in that channel | genuine result, not an error |
| cross-target spread ≫ reported σ | systematic FK bias | expected; quote the spread |

---

## 8. Limits, and what would actually improve accuracy

Reweighting downweights `A` correctly and yields honest uncertainty, but **it cannot
remove systematic error** — the bias stays in `X` and `Y`.

The Aug 25 data shows the mechanism directly. FK says `link_torso_5` moved 0.004 mm
across the whole capture; mocap says it moved 0.379 mm rms, up to 1.878 mm — 16×
its own noise floor, and repeatable per arm configuration (0.018 mm within a
configuration, 0.379 mm between). **The torso flexes under the arm**, FK is blind to
it because the torso joints never moved, and every arm link is mounted on that
torso.

Two ways forward:

1. **Re-reference to the physical torso.** Solve `A'X = Y'B'` with
   `A' = torso_T_link` (the arm chain) and `B' = rigid_torso_T_rigid_link`
   (mocap-to-mocap). Torso flex then cancels on both sides — worth ~2 mm. The NPZ
   files carry `original_window_index` for the cross-target matching this needs.
2. **Kinematic parameter identification.** The real fix for the rest. This dataset —
   six links plus the end effector, mocap-observed across twenty configurations — is
   close to what that requires.

---

## Reference

| file | role |
|---|---|
| `examples/calibrate_rby1.py` | the calibration workflow |
| `probabilistic_axyb/initialization.py` | closed-form init, excitation check |
| `probabilistic_axyb/solver.py` | ML solvers, configurations 1/2/3 |
| `probabilistic_axyb/uncertainty.py` | calibration covariance |
| `calibration_report.html` | Aug 25 results, with column definitions |
| `README.md` § *Choosing a noise model on real data* | the reasoning behind §5 |

## 9. How much data the calibration needs

Measured on the Aug 27 end effector (500 configurations x 3 repeats), calibrating on a
random subset and comparing `X`, `Y` against the full-data solution:

| configs used | poses | `Y` shift | `X` shift |
|---|---|---|---|
| 25 | 75 | 2.25 mm / 0.09 deg | 1.46 mm / 0.43 deg |
| 50 | 150 | 1.52 mm / 0.09 deg | 0.87 mm / 0.07 deg |
| **100** | 300 | **0.69 mm / 0.04 deg** | 0.25 mm / 0.13 deg |
| 200 | 600 | 0.95 mm / 0.04 deg | 0.45 mm / 0.06 deg |
| 500 | 1500 | reference | reference |

Below ~1.5 mm the shift is under the cross-calibration floor and cannot be told
from noise: **~100 well-spread configurations saturate the calibration.** The
labels, by contrast, need every pose you intend to train on -- but a label is a
matrix product per pose, not a solve. `build_error_dataset.py --calibrate-on 100`
solves on 100 random configurations and applies `X`, `Y` to all poses; the full
solve remains the default.
