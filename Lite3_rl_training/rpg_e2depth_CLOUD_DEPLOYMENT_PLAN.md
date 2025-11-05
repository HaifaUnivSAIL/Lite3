# Cloud Deployment Plan for rpg_e2depth Sweeps

## Executive Summary
- **Workload profile (from the paper):** recurrent UNet-style network operating on event-voxel grids; trained with scale-invariant + multi-scale gradient losses; batch size≈20; supervised on DENSE (CARLA) and finetuned/evaluated on MVSEC. This is a **single-GPU friendly** model—modern 24 GB GPUs (e.g., 4090/L40S/A10) are ample.
- **Strategy:**
  - **Tier 1 (cheap, wide sweeps):** Vast.ai + RunPod (1× RTX 4090 class).
  - **Tier 2 (stable, reproducible long runs):** Lambda Cloud (A10/A6000/A100/H100) + Google Cloud (L4).
- **Headline costs today (on-demand list prices):**
  - Vast.ai **RTX 4090 ~$0.29/hr** (P25 reference)
  - RunPod **RTX 4090 ~$0.34–$0.59/hr**
  - Lambda Cloud **H100 $2.49/hr**, cheaper SKUs (A10/A6000/A100) much less
  - Google Cloud **L4 ~$0.7–$0.8/hr**
  - AWS **g6e.xlarge (L40S)** **$1.86/hr** (for reference)

---

## 1. What the Model Needs (Practical)
- **Model complexity:** recurrent fully-convolutional UNet with ConvLSTMs; voxelized events with e.g. B temporal bins; unrolled a few steps; trained with SI + multi-scale gradient losses. Batch size reported as **20**.
- **Data footprint:** DENSE (synthetic) + MVSEC (real). Expect **tens of GB**, not TB. Stage them once to object storage; each worker syncs a subset/cache.
- **Scaling style:** rather than bigger GPUs, we scale **horizontally** with many cheap 1-GPU workers for parallel W&B runs.

---

## 2. The Deployment Plan (Two Tiers)

### Tier 1 — Broad Sweeps, Ultra-Low Cost
**Providers:** **Vast.ai** and **RunPod** (RTX 4090 class)

**Why these:** Lowest $/GPU-hr for bursty, many-short-runs workflows; easy Docker; great for 30–200 parallel W&B runs.

**Current reference pricing**
- Vast.ai: RTX 4090 **~$0.29/hr (P25)**; H100 ~$1.65/hr.
- RunPod: RTX 4090 **$0.34–$0.59/hr** depending on availability.

**Usage plan:**
- One unified Docker image (PyTorch + CUDA, repo + entrypoint).
- W&B key and optional S3/GCS credentials passed as environment variables.
- Datasets fetched from object storage; cached locally for multi-hour runs.
- Jobs short-lived (1–3 h), frequent checkpointing to W&B.

**Cost example:**
- 100 runs × 2 h = **200 GPU-hrs**
- At $0.30/hr → **$60 total**

---

### Tier 2 — Stable, Reproducible, Longer Jobs
**Providers:** **Lambda Cloud** (A10/A6000/A100/H100) and **Google Cloud (L4)**

**Why these:** predictable environments, clean networking/VPC integration, better SLAs, easier “final run” reproducibility and internal sharing.

**Current reference pricing**
- Lambda Cloud: H100 **$2.49/hr**; cheaper SKUs (A10/A6000/A100) much less.
- Google Cloud: L4 **~$0.7–$0.8/hr** depending on region.

**Usage plan:**
- Long 12–24h training runs, deterministic seeds, full W&B logging.
- Use GCS buckets for dataset and checkpoints.
- Optionally use spot/preemptible for lower cost if checkpointing enabled.

**Cost example:**
- 2 jobs × 24 h = 48 GPU-hrs
- At $0.75/hr → **$36 total**

---

## 3. Future Work to Make Everything Actually Run

### A) Containerization & Reproducibility
- Single Dockerfile pinned to CUDA base image (e.g., `pytorch/pytorch:2.x-cuda12.x-cudnn8-runtime`)
- Install: PyTorch, kornia, OpenCV, W&B, rpg_e2depth dependencies.
- Entrypoint executes training command using environment parameters.
- Healthcheck ensures CUDA availability.

### B) Data Plumbing
- Immutable dataset bucket (S3/GCS): `s3://<bucket>/datasets/{DENSE,MVSEC}/...`
- Startup sync script: verify cache, download missing splits, checksum verify, log to W&B Artifacts.

### C) W&B Sweeps at Scale
- Sweep YAML with hyperparameters (B, unroll, LR, optimizer, augment flags, loss weights).
- Use `wandb agent <entity>/<project>/<sweep_id>`.
- Frequent checkpoints (every 5–10 mins) for preemption safety.

### D) Secrets & Config
- Pass W&B API key, cloud keys via environment variables or secret managers.

### E) Cost Guardrails
- Auto-terminate idle instances.
- Cap max parallelism (e.g., 30 GPUs on Vast.ai).
- W&B org-level budget alerts.

### F) Observability
- W&B metrics for GPU/CPU utilization, VRAM, and I/O.
- Push logs and configs as artifacts per run.

---

## 4. Concrete Runbooks

### Tier 1 (Vast.ai / RunPod)
1. Build & push image to Docker Hub or GHCR.
2. Startup script:
   ```bash
   docker pull <image>
   export WANDB_API_KEY=<key>
   bash start.sh
   ```
3. Provision:
   - Vast.ai: RTX 4090 offers ≥24 GB VRAM, reliability filter, price ≤$0.40/hr.
   - RunPod: choose 4090 Community/Pro pod, persistent volume optional.
4. Scale: launch N workers; W&B agent handles job queueing.

### Tier 2 (Lambda Cloud / Google Cloud)
1. Lambda Cloud:
   - Choose A10/A6000/A100 for final experiments.
   - On-demand for long 12–24h training.
2. Google Cloud:
   - MIG template with L4 (~$0.7/hr).
   - Store datasets & checkpoints in GCS.
   - Use same Docker image and entrypoint as Tier 1.

---

## 5. Costed Examples

| Scenario | Provider/GPU | Hours | $/hr | Total |
|-----------|---------------|------:|------:|------:|
| Wide sweep: 100×2h | Vast.ai RTX 4090 | 200 | 0.30 | $60 |
| Wide sweep: 100×2h | RunPod RTX 4090 | 200 | 0.45 | $90 |
| Final train: 2×24h | GCP L4 | 48 | 0.75 | $36 |
| Final train: 2×24h | Lambda A10/A6000 | 48 | 1.00 | $48 |

---

## 6. Minimal Tech Stack Checklist

**Week 1 — Make it run anywhere**
- [ ] Dockerfile (CUDA base, PyTorch, kornia, OpenCV, W&B, repo)
- [ ] `start.sh` (auth, dataset sync, run agent)
- [ ] W&B sweep YAML + default config
- [ ] Bucket for DENSE+MVSEC datasets
- [ ] Local test run on 24 GB GPU

**Week 2 — Tier 1 at Scale**
- [ ] Vast.ai template (best-price filters)
- [ ] RunPod template (community/pro, persistent volume)
- [ ] Parallel launch script with spend cap
- [ ] W&B artifact tracking

**Week 3 — Tier 2 for Finals**
- [ ] Lambda on-demand instances (A10/A6000)
- [ ] GCP L4 setup (Artifact Registry + GCS)
- [ ] Preemptible policy + resume logic
- [ ] Repro checklist (seed, commit, image digest)

---

## 7. Appendix — Paper Details Justifying the Setup
- Recurrent UNet with ConvLSTM encoder, voxelized event input.
- Batch size: 20; training losses: scale-invariant + multi-scale gradient.
- Pretrain on CARLA/DENSE; finetune on MVSEC.
- Fits easily on one modern GPU (24 GB VRAM).

---

## Sources
- RPG Paper (Hidalgo et al., 3DV 2020): https://arxiv.org/abs/2010.08350
- Vast.ai pricing pages
- RunPod pricing/model pages
- Lambda Cloud on-demand pricing
- Google Cloud GPU pricing (official)
- AWS g6e.xlarge (L40S) reference pricing
