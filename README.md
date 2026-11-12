# Annotation assistant — project 3

Status: runnable binary segmentation MVP: a small PyTorch student, seeded training,
validation-selected checkpoints, held-out testing, pool ranking, a review interface,
provenance records, and import into a new training-round manifest.

Goal: reduce human annotation minutes required to train a domain segmentation model.
Teacher reference: https://github.com/facebookresearch/sam3 (checkpoint access required).

`student.py` generates student features and pixel entropy for active selection.
`selection.py` combines uncertainty and embedding diversity. SAM teacher integration
is still planned; the current mask suggestions come from the trained student.

## Quick start

Create an independent environment and install `requirements.txt`. In my workspace,
my existing sibling `../hybrid-rag/.venv` also works; it contains CPU-only PyTorch.

```powershell
python make_demo.py
python student.py train data/demo/manifest.json --output runs/demo --epochs 5
python student.py evaluate data/demo/manifest.json --checkpoint runs/demo/student.pt
python student.py rank data/demo/manifest.json --checkpoint runs/demo/student.pt --budget 4 --strategy combined
python -m streamlit run app.py
```

Use `random`, `uncertainty`, `diversity`, or `combined` to compare selection strategies.
Rank output contains pool IDs, paths, entropy, and strategy. The current review UI lets
me choose any pool image; refer to selection.json for the suggested order. It shows
the student's binary mask and accepts a corrected PNG with the original dimensions.
I must explicitly approve a reviewed mask before saving. Saving writes a mask and
audit JSON under annotations/; it never changes the manifest or silently adds labels.

```powershell
python import_reviews.py data/demo/manifest.json --output data/round-2.json
python student.py train data/round-2.json --output runs/round-2 --epochs 10
```

Import refuses duplicate reviews and non-pool samples, verifies reviewed image hashes,
and writes a new manifest with reviewed samples promoted to training. Archive old
review records before subsequent rounds so only the current round is imported.
Validation and test splits are unchanged. Imported manifests use absolute local paths;
do not publish them if they contain private image paths.

## Review integrity update

Approval is bound to the current image, checkpoint and accepted mask. Changing any
of those inputs clears approval. Version-2 review records include image, mask and
checkpoint SHA-256 values. Import rejects a changed mask or image, nonbinary masks,
invalid review times and old records without these checks. For a legacy review,
reopen the image and approve its mask again; archive the old review before import.
The checkpoint hash is preserved as provenance; the original checkpoint does not
need to remain available to import an otherwise valid review.

An optional `group` field identifies an acquisition session/physical object.
If supplied, the same group cannot cross train/validation/test/pool splits.
Define groups before collecting labels to avoid promoting related test views.

## Custom data and protocol

A manifest is a JSON list with unique `id`, relative `image`, `split`, and optional
`mask` fields. Splits: train, validation, test, pool. Training/validation/test labels
are binary PNG masks: zero is background, nonzero is foreground. Pool labels are absent.
Exact image content shared across splits is rejected; separate acquisition sessions
yourself to prevent near-duplicate leakage. Student inputs are resized to 96 x 96.

Training uses BCE and selects the best validation IoU checkpoint. Test labels are
never used for selection. Evaluation reports dataset-level pixel IoU and mean BCE.
The tiny student is a deliberately simple baseline, not a pretrained foundation model.
Synthetic red-circle images validate the pipeline, not real-world segmentation quality.
Review time measures elapsed UI time and may include breaks; it is not precise active
labor time. Human approval does not guarantee label correctness.

The initial diversity implementation starts from the most uncertain image, then uses
farthest-first cosine distance; label it as uncertainty-seeded diversity in experiments.
No claims of measured annotation-time savings are made. Comparing complete annotation
rounds at equal human-time budgets and integrating a teacher are next milestones.

```powershell
python -m unittest -v
```

## Build order

1. Choose one domain and independently label a held-out test set.
2. Build a small PyTorch segmentation student and a reproducible training loop.
3. Add teacher mask suggestions and human correction, preserving label provenance.
4. Compare random, uncertainty, diversity, and combined selection with fixed seeds.
5. Track IoU and boundary quality versus actual annotation minutes, plus inference cost.

Use equal human-time budgets. Teacher predictions are suggestions, not ground truth.
Keep test images out of active selection and model tuning. Record corrections and time
per image. On my 8 GB GPU, run teacher and student sequentially.

## Docker image

This project has its own CPU image: `ghcr.io/khawajasaad22/annotation-assistant:latest`.
Images are stored in GitHub Container Registry; the repository contains the Dockerfile,
Compose configuration, and build/publish workflow. The workflow tests the image and
checks the running app before publishing `latest` and a full commit-SHA tag. A failed
build does not publish. Use the Actions tab to confirm the first successful publication.

With Docker Engine/Desktop and Compose installed, from this repository:

```sh
docker compose up --build -d
```

Open http://localhost:8504. To use the registry image after it is published:

```sh
docker compose pull app
docker compose up -d --no-build app
```

If the registry package is private, authenticate with `docker login ghcr.io` using an
account/token allowed to read the package. Repository visibility alone does not ensure
anonymous image access. No custom registry secret is needed by the publishing workflow:
it uses the repository's built-in GitHub token with package-write permission.

The container runs as UID 10001. Named volumes preserve inputs, generated artifacts,
and downloaded model weights. Weights and private datasets are not baked into the image.
First model use requires internet access. These are Linux/amd64 CPU images; GPU/CUDA
acceleration is not included. Initial builds can take several minutes.

Use `docker compose exec app python ...` for the documented CLI commands. Inputs are
container paths, not Windows host paths. Copy real files with `docker compose cp`:

```sh
docker compose cp ./my-input app:/app/data/my-input
docker compose logs -f app
docker compose down
```

`down` keeps named volumes; `down -v` deletes them, including my generated work.
The app port is published on localhost only. Container health means Streamlit is running;
I still need to prepare data and fit/index models before using the app workflow.

```sh
docker compose exec app python make_demo.py
docker compose exec app python student.py train data/demo/manifest.json --output runs/demo --epochs 5
docker compose exec app python student.py rank data/demo/manifest.json --checkpoint runs/demo/student.pt --budget 4 --output artifacts/selection.json
```

Reviews persist in the annotations volume; checkpoints persist in runs. The demo and
run generators refuse overwrites: reuse existing outputs or choose a fresh run folder.
