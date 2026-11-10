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

Create an independent environment and install `requirements.txt`. In this workspace,
the existing sibling `../hybrid-rag/.venv` also works; it contains CPU-only PyTorch.

```powershell
python make_demo.py
python student.py train data/demo/manifest.json --output runs/demo --epochs 5
python student.py evaluate data/demo/manifest.json --checkpoint runs/demo/student.pt
python student.py rank data/demo/manifest.json --checkpoint runs/demo/student.pt --budget 4 --strategy combined
python -m streamlit run app.py
```

Use `random`, `uncertainty`, `diversity`, or `combined` to compare selection strategies.
Rank output contains pool IDs, paths, entropy, and strategy. The current review UI lets
you choose any pool image; refer to selection.json for the suggested order. It shows
the student's binary mask and accepts a corrected PNG with the original dimensions.
You must explicitly approve a reviewed mask before saving. Saving writes a mask and
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
per image. On the 8 GB GPU, run teacher and student sequentially.
