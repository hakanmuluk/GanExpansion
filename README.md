# GanExpansion

**Generative policies with independent keepout and waypoint skill adapters.**

GanExpansion studies generative policies and independent skill addition in two environments: continuous **TwoRoom navigation** and native **OGBench single-cube manipulation**. This guide covers base models and separately trained keepout and waypoint adapters. The skill results and examples evaluate one enabled adapter at a time.

The repository contains source, tests, configurations, and this guide. Pretrained weights and benchmark inputs are available through the [v1.0.0 release](https://github.com/hakanmuluk/GanExpansion/releases/tag/v1.0.0). The download script installs them into the paths expected by the original code.

## Quick start

**Recommended:** a Linux machine or Colab runtime with a GPU, Python 3.10+, and an existing CUDA-enabled PyTorch installation. Python 3.11 is a convenient choice for a new environment. The TwoRoom examples also work on CPU.

~~~bash
git clone https://github.com/hakanmuluk/GanExpansion.git
cd GanExpansion

# Keep Colab's existing PyTorch. For a new environment, install the appropriate
# PyTorch build first: https://pytorch.org/get-started/locally/
python scripts/setup_environment.py --env all
python scripts/download_assets.py --bundle tworoom
python scripts/download_assets.py --bundle cube

# Original generative policy.
python examples/run.py --env tworoom --case base --device cpu

# Enable one independently trained skill.
python examples/run.py --env tworoom --case keepout --device cpu
~~~

Setup preserves the installed PyTorch build. It installs the minimal dependencies for these tasks, including stable-worldmodel from source at commit **21446f1ede6d5284e981bd7b47f432b994e6d812**. The PyPI package sharing version 0.1.1 has a different environment API; use the pinned source.

For TwoRoom only:

~~~bash
python scripts/setup_environment.py --env tworoom
python scripts/download_assets.py --bundle tworoom
~~~

Native Cube uses MuJoCo 3.11.0, OGBench 1.2.2, and dm_control 1.0.44. Linux headless rendering uses EGL. The historical GPU measurements used A100/L4 GPUs; exact numerical and timing results can vary with hardware and library versions.

## What was trained?

### 1. Base generative policies

Both policies predict **16 future actions and execute eight before replanning**, with a two-dimensional Gaussian behavior code held fixed throughout an episode.

| | TwoRoom | Cube |
|---|---|---|
| Observations | 10 state features | 47 state features |
| Predicted controls | 16 × 2 | 16 × 5 |
| Original code controls | Door, crossing offset, speed | Carry side, lateral amplitude, speed |
| Training | Scripted successful trajectories | Actual native robot demonstrations |

The architecture is a style-modulated residual generator, a masked whole-episode encoder, and a conditional discriminator. It uses BicycleGAN-inspired reconstruction and latent-cycle objectives, KL regularization, and adversarial training.

Both policies take numerical simulator states as input. Rendered images and videos are used for visualization.

The strongest selected recipes also use **known synthetic expert style labels, additional reconstruction targets, and feedback/recovery supervision**. Their success should not be attributed to vanilla BicycleGAN alone. The generator produces the deployed actions; the base policy does not call an expert at inference time.

### 2. Independent skill adapters

Each new skill has its own Gaussian modifier code, mapping, low-rank weight updates, conditional scale/shift modulation, encoder, discriminator, and binary enable switch:

- **Keepout:** reach the original goal while avoiding an additional forbidden region.
- **Waypoint:** visit a required region and then finish the original goal.

TwoRoom modifiers choose detour/approach sides and their extent. Cube modifiers choose different vertical bypass/approach strategies while carrying the cube. The original behavior code remains available alongside the new code.

The base generator, original encoder, and normalization remain frozen. Disabling a skill invokes the preserved base path exactly. Active skills may change controls needed to complete the added task; the preservation loss encourages small corrections rather than guaranteeing local changes.

## Tasks and benchmark definitions

| Benchmark | What success means |
|---|---|
| TwoRoom base | Navigate through either door and reach an unseen goal under the upstream collision convention. |
| TwoRoom individual skills | Original goal plus swept keepout avoidance or an observed waypoint visit. |
| Cube base | Native single-cube goal placement; released-and-settled placement is reported separately from entering the goal radius. |
| Cube individual skills | Released/settled placement plus the carried-cube constraint and supported transport. |

TwoRoom uses stable-worldmodel's continuous environment, not a discrete MiniGrid TwoRoom task. Cube runs the actual native MuJoCo/OGBench robot. The added keepout/waypoint tasks are **custom constrained benchmark distributions**, not official aggregate OGBench scores.

The Cube keepout region is virtual and covers the transported cube, using its conservative complete-body bounding sphere. It is not an additional native collision body and does not specify whole-arm avoidance. Waypoint entry requires the cube to be held. Geometry is sampled at native 2 ms steps; support/contact diagnostics use 50 ms controls. The native goal tolerance remains 4 cm.

## Success rates

### Base policies

**TwoRoom:** goal success on 256 unseen start/goal pairs with 32 independently sampled Gaussian codes per pair (8,192 episodes per seed).

| Training seed | Goal success |
|---|---:|
| 42 | 100.00% |
| 123 | 100.00% |
| 2026 | 100.00% |

**Cube:** 150 held-out simulator contexts × eight Gaussian codes (1,200 native episodes per seed). Native goal success means entering the 4 cm goal radius. Released-and-settled success additionally requires placement, release, low cube speed, and end-effector clearance for ten consecutive controls.

| Training seed | Native goal success | Released and settled |
|---|---:|---:|
| 42 | 97.17% | 95.50% |
| 43 | 97.58% | 95.33% |
| 44 | 96.17% | 93.75% |

### Independent TwoRoom adapters

Each adapter is tested on 256 unseen contexts × eight Gaussian modifier codes (2,048 episodes per seed). **Full skill success requires the original goal and the enabled adapter's constraint.**

| Model / adapter training seed | Keepout full skill | Waypoint full skill |
|---|---:|---:|
| Frozen base, no adapter | 0.00% | 0.00% |
| Adapter, seed 42 | 99.32% | 91.16% |
| Adapter, seed 123 | 99.41% | 97.02% |
| Adapter, seed 2026 | 99.56% | 98.00% |
| Adapter mean ± seed SD | **99.43% ± 0.12 pp** | **95.39% ± 3.70 pp** |

The base still reaches its original goal in 100.00% of these skill-bank episodes. The new banks deliberately place keepout on its original route or waypoint away from that route, so it fails the added constraint without an adapter. The waypoint metric permits early goal-region entry before eventual waypoint completion; it does not require the waypoint visit before first goal entry. Native TwoRoom collision success also differs from a strict circular-body wall-overlap check.

### Independent Cube adapters

Each adapter is tested on 128 unseen contexts × eight paired Gaussian modifier codes (1,024 native episodes per seed). **Full skill success requires released-and-settled placement plus the enabled constraint and supported cube transport.**

| Model / adapter training seed | Keepout full skill | Waypoint full skill |
|---|---:|---:|
| Frozen base, no adapter | 0.00% | 0.00% |
| Adapter, seed 42 | 92.48% | 93.46% |
| Adapter, seed 43 | 95.21% | 95.70% |
| Adapter, seed 44 | 96.39% | 95.41% |
| Adapter mean ± seed SD | **94.69% ± 2.00 pp** | **94.86% ± 1.22 pp** |

The frozen base achieves 88.28% released placement on these added-skill banks but 0.00% full added-skill success. These banks differ from the original base benchmark above, so the placement percentages are not a measure of forgetting.

Here **pp** means percentage points; SD is the sample standard deviation across three adapter training seeds. Seeds share held-out context identities, so repeated model executions are not independent scene draws. All adapter rates above concern the named skill on its own benchmark.

## Run each case

Run commands from the cloned repository root. Examples default to **four contexts and two codes**. These are small demos, not the complete test tables.

### TwoRoom

~~~bash
# Original policy: sample an episode-level behavior code.
python examples/run.py --env tworoom --case base --device cpu

# Enable one independent adapter.
python examples/run.py --env tworoom --case keepout --device cpu
python examples/run.py --env tworoom --case waypoint --device cpu
~~~

### Cube

~~~bash
# Original native manipulation policy.
python examples/run.py --env cube --case base --device cpu

# Original independent scalar adapters.
python examples/run.py --env cube --case keepout --device cpu
python examples/run.py --env cube --case waypoint --device cpu
~~~

Examples default to model seed 42. Override with --model-seed: TwoRoom supports 42/123/2026, and Cube supports 42/43/44. The --seed argument controls sampled behavior/modifier codes and defaults to 42.

Use --count and --codes to change demo sizes, --out for a fresh output folder, and --dry-run to inspect model/input choices without importing ML libraries. Existing nonempty output folders are rejected.

~~~bash
python examples/run.py --env cube --case waypoint \
  --count 16 --codes 8 --model-seed 43 --device cpu \
  --out outputs/cube_waypoint_s43
~~~

Each example saves summary.json and trajectories or per-episode records under outputs/. This directory is excluded from Git.

## Weights and reproducible inputs

| Release bundle | Compressed size | Contents |
|---|---:|---|
| tworoom-models-v1.zip | 17.3 MB | Three base models, selected skill models for three seeds, original discriminator initialization |
| cube-models-v1.zip | 29.3 MB | Three base models, six original skill adapters, original discriminator initialization |
| benchmark-inputs-v1.zip | 19.1 MB | Frozen evaluation inputs and supporting metadata |

~~~bash
python scripts/download_assets.py --bundle tworoom
python scripts/download_assets.py --bundle cube
python scripts/download_assets.py --bundle tworoom --verify-only
python scripts/download_assets.py --bundle cube --verify-only
~~~

The downloader verifies archive/member SHA-256 checksums, rejects unsafe paths, skips already verified files, and resolves dependencies. Downloads go into the original artifacts-style paths, which remain ignored by Git. Modified files require an explicit --force before replacement.

The versioned manifest is [configs/assets.json](configs/assets.json). Original inference checkpoint bytes are preserved because adapters verify the base checkpoint fingerprint. A newly trained base needs newly trained compatible adapters; it cannot silently replace the published base.

Large original optimizer/RNG resume archives and research-result archives are omitted. Compact selected checkpoints retain their original fields and hashes; the examples use them for inference or adapter initialization. The two discriminator.pt files contain original discriminator tensors for adapter initialization and are **not optimizer-resume checkpoints**.

## Full evaluation and inspection

The example runner delegates to the original scientific implementations. Use their module CLIs for full evaluations and ablations.

~~~bash
# Original TwoRoom base benchmark.
python -m tworoom_gan.evaluate \
  --checkpoint artifacts/final/base_policy.pt --data artifacts/data_v2/test.npz \
  --pairs 256 --latents 32 --seed 9001 --out outputs/tworoom_base_full

# Original native Cube base benchmark. This evaluator uses CPU model workers.
python -m cube.cube_gan.evaluate \
  --checkpoint cube/artifacts/final/base_policy.pt --data cube/artifacts/data \
  --contexts 150 --latents 8 --workers 6 --out outputs/cube_base_full

# Original individual skills, including preservation and encoder diagnostics.
python -m tworoom_addons.evaluate --skill keepout \
  --checkpoint addon_artifacts/final/keepout.pt --codes 8 \
  --out outputs/tworoom_keepout_full
python -m cube_addons.evaluate --skill waypoint \
  --checkpoint cube_addon_artifacts/final/waypoint_s42/adapter.pt \
  --contexts cube_addon_artifacts/data_force/waypoint/test_contexts.npz \
  --codes 8 --seed 20261122 --workers 6 --force-contacts \
  --out outputs/cube_waypoint_full
~~~

Plotting and native replay utilities are included for the base policies and independent skills. Plots and videos are generated locally.

## Train new models

Training requires a GPU for the reported recipes. Data-generation scripts, experts, training losses, and feedback collectors are included. Training outputs are local and ignored by Git.

**Base policies:** collect new training data, then train. Defaults provide a starting experiment; the auxiliary flags below are important for the stronger TwoRoom recipe.

~~~bash
python -m tworoom_gan.data --out outputs/tworoom_data --route-offset 12
python -m tworoom_gan.train --run outputs/tworoom_base --data outputs/tworoom_data \
  --steps 5000 --seed 42 --batch 192 --cycle-batch 8 \
  --encoder-moments --canonical --route-feature \
  --style-weight 2 --prior-rec-weight 10 --latent-weight .1 \
  --adv-weight .05 --kl-weight .005 --augment-pixels 4 \
  --rollout-cycle-weight .5 --rollout-cycle-every 8 --rollout-cycle-start 2000 \
  --feedback-weight 5 --feedback-every 8 --feedback-start 1000

python -m cube.cube_gan.data --out outputs/cube_data --workers 6
python -m cube.cube_gan.train --run outputs/cube_base --data outputs/cube_data \
  --steps 4000 --seed 42 --route-feature --style-weight 2 --known-rec-weight 10
~~~

The published Cube base also includes encoder-only calibration; a fresh basic run is not a guaranteed reproduction of the selected table. Training configurations for the published models are in [configs/training](configs/training). They identify historical extra feedback inputs and budgets.

**Original adapters:** keep the published base, generate single-skill data, then train separate models:

~~~bash
python -m tworoom_addons.data --out outputs/tworoom_skill_data
python -m tworoom_addons.train --skill keepout --data outputs/tworoom_skill_data \
  --run outputs/tworoom_keepout --steps 3000 --seed 42
python -m tworoom_addons.train --skill waypoint --data outputs/tworoom_skill_data \
  --run outputs/tworoom_waypoint --steps 3000 --seed 42 \
  --rank 8 --context-hidden 128 --modulation-rank 16 --output-modulation \
  --feedback-weight 12 --feedback-every 8 --tracking-weight 5

python -m cube_addons.data --out outputs/cube_skill_data --workers 6
python -m cube_addons.train --skill keepout --data outputs/cube_skill_data \
  --run outputs/cube_keepout --steps 4000 --seed 42 \
  --rank 8 --output-modulation --lr .0005 --separate-clip
~~~

These fresh-training examples are runnable starting recipes. Exact original Cube skill results additionally used training-only learner-state corrections; waypoint used planning-aligned demonstrations. Those are implemented in cube_addons.feedback, cube_addons.data --macro-waypoint, and the saved training configurations. Report the extra supervision when comparing data budgets.

## Code map and tests

| Package | Purpose |
|---|---|
| tworoom_gan | Base TwoRoom policy, demonstrations, GAN/encoder training |
| tworoom_addons | Independent TwoRoom skills and preservation |
| cube/cube_gan | Native Cube base policy and training |
| cube_addons | Native skills, force support, feedback, encoders |

~~~bash
# After setup and downloading the base/adapter assets:
python -m pytest -q tests addon_tests cube_tests cube_addon_tests
~~~

These tests cover frozen weights, exact disabled behavior, geometry/dynamics parity, and encoder behavior. Scientific source, checkpoints, and release assets are preserved.

The project uses state observations, custom constraint distributions, and simulated robots. It does not establish visual navigation, new room-layout generalization, whole-robot collision avoidance, or real-world robot safety. Keep success definitions and inference cost alongside reported percentages.

## Attribution and license status

Environment code comes from [stable-worldmodel](https://github.com/galilai-group/stable-worldmodel) and [OGBench](https://github.com/seohongpark/ogbench), with [MuJoCo](https://github.com/google-deepmind/mujoco) physics. The architectures draw on BicycleGAN, StyleGAN, and LoRA. See [THIRD_PARTY_NOTICES.txt](THIRD_PARTY_NOTICES.txt) for references and the retained upstream license.

No project-wide license for the original code or trained weights has been selected.
