# Reproduction of GenSafeNav (evaluation with the authors' checkpoint)

Paper: *Towards Generalizable Safety in Crowd Navigation via Conformal Uncertainty Handling*
(Yao et al., CoRL 2025, [arXiv:2508.05634](https://arxiv.org/abs/2508.05634)).
Original code: [tasl-lab/GenSafeNav](https://github.com/tasl-lab/GenSafeNav) (MIT).

This folder documents my attempt to reproduce the paper's results. The code outside this
folder is the unmodified upstream fork.

**Status:** the evaluation of the authors' released checkpoint is reproduced.
Training from scratch and the out-of-distribution experiments (Table 2) are **not done yet**
(see [Not done](#not-done-and-why)).

## Main result

Model `Ours_GST`, checkpoint `05207.pt`, in-distribution setting, `python test.py`.

| Metric | My run | Authors' log in repo | Paper, Table 1 (mean ± std) |
|---|---|---|---|
| Success rate (SR) | **97.20%** | 97.20% | 96.93 ± 0.68% |
| Collision rate (CR) | **2.80%** | 2.80% | 2.93 ± 0.61% |
| Timeout rate (TR) | 0.00% | 0.00% | 0.13 ± 0.12% |
| Navigation time (NT) | 17.74 | 17.69 | 17.54 ± 0.86 |
| Path length (PL) | 24.97 | 24.91 | 24.27 ± 0.85 |
| Intrusion time ratio (ITR) | 2.91% | 2.96% | 2.72 ± 0.16% |

- My run and the authors' log report the **same seven collision episodes**: 94, 95, 105, 130, 141, 221, 226.
- **Protocol caveat.** `test.py` hard-codes `test_size = 250` (line 157; the config value 500 is not used), i.e. one test seed.
  The paper's Table 1 averages 1250 samples (5 test seeds x 250) over models trained with 3 training seeds.
  So my number is consistent with the paper but is **not** the same statistic.
- Small differences in NT/PL/ITR versus the authors' log are probably due to library versions
  (Python 3.13, PyTorch 2.14, NumPy 2.x here versus the authors' environment); I did not verify the cause.

Raw logs are in [`results/`](results/).

## Observation: the 7 collisions

Episode length is in simulation steps (parsed from `results/stdout_Ours_GST_250ep.log`; mean over all 250 episodes is 71 steps).

| Episode | Length (steps) | Final reward |
|---|---|---|
| 94 | 23 | -15.6 |
| 95 | 42 | -7.9 |
| 105 | 9 | -20.0 |
| 130 | 42 | -10.3 |
| 141 | 107 | -5.5 |
| 221 | 30 | -12.5 |
| 226 | 11 | -19.9 |

Six of the seven collisions happen within the first 42 steps, two of them within 11 steps.
This suggests collisions come mostly from the start of an episode rather than from the middle
of a long path. I have **not** inspected the initial configurations, so the cause is unknown;
this is a lead for further analysis, not a finding.

## How to reproduce

Open `reproduction.ipynb` in Google Colab (T4 GPU) and run it top to bottom (about 30 minutes;
the 250 episodes took about 23 minutes because the simulator is CPU-bound).

Environment used: Colab, Tesla T4, Python 3.13.16, PyTorch 2.14.1+cu130.
The repo's Dockerfile targets PyTorch 2.3.1 / CUDA 12.1; I did not use Docker.

## Issues and fixes

| Problem | Fix |
|---|---|
| `Python-RVO2` fails to build under pip build isolation | `pip install Cython setuptools wheel cmake`, then `pip install --no-build-isolation .` |
| `torch.load` fails: `Weights only load failed` (PyTorch >= 2.6 default `weights_only=True`; the checkpoint contains a NumPy scalar) | `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`. Only do this for the authors' own, trusted checkpoints. |
| `test.py` overwrites the authors' log `trained_models/Ours_GST/test/test_05207.pt.log` (opened with mode `w`) | Copy `trained_models/` somewhere before the first run (done in the notebook). |
| `python test.py --visualize --test_case 94 ...` fails with `unrecognized arguments` | Cause: `get_args()` in `arguments.py` calls `parser.parse_args()` on the same `sys.argv`. Patching it to `parse_known_args()[0]` removes the error. |
| Replay of episode 94 produced no output within 10 minutes (timeout) | Not resolved; reason unknown. The replay is not part of the main result. |

## Not done, and why

- **Training from scratch (`train.py`).** The paper (Appendix E) trains for 2 x 10^7 steps with
  128 parallel environments on an RTX 4090, with 3 training seeds. In Colab the simulator ran on a single
  CPU core, so I estimate days of runtime for one seed (my own rough estimate, not measured on training).
  A smaller-scale run or a multi-core machine is needed.
- **Out-of-distribution experiments (Table 2)**, and multiple test seeds.
- **Visualization** of individual episodes (see the last row above).

## Next steps

1. Decide with the supervisor between a reduced-scale training run and access to a larger machine.
2. Run more test seeds to match the paper's 1250-sample protocol.
3. Inspect the initial conditions of the seven collision episodes.
