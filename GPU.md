# GPU execution

For my RTX 4060, use the optional Linux GPU configuration:

```sh
docker compose -f compose.yaml -f compose.gpu.yaml up --build -d
docker compose -f compose.yaml -f compose.gpu.yaml exec app python gpu_check.py
```

This builds a distinct local CUDA-capable image instead of pulling the published
CPU image. It requests NVIDIA GPU passthrough and fails startup if CUDA computation
is unavailable. The base Dockerfile remains CPU-compatible for ordinary CI/hosting.
Windows requires the Docker/WSL2 NVIDIA GPU setup and a compatible host driver.

The override installs Linux PyTorch from PyPI rather than the CPU-only wheel index.
That can download several GB of CUDA dependencies. The selected wheel/runtime must
match the host driver; supplying a GPU alone does not change a CPU-only PyTorch build.
Versions remain compatibility ranges, so record/lock the validated deployment stack.

Training, evaluation and ranking accept `--device auto|cpu|cuda`. Auto selects CUDA
when available. Explicit cuda rejects a missing GPU instead of silently using CPU.
Model batches move onto the selected device; predictions/embeddings return to CPU
for NumPy/selection. Checkpoints save CPU tensors and can load on either device.

```sh
docker compose -f compose.yaml -f compose.gpu.yaml exec app python student.py train data/demo/manifest.json --output runs/cuda-demo --epochs 5 --device cuda
docker compose -f compose.yaml -f compose.gpu.yaml exec app python student.py evaluate data/demo/manifest.json --checkpoint runs/cuda-demo/student.pt --device cuda
```

Create the synthetic demo first if it is not present. Use a new run directory.
Training history records the selected device, framework/runtime, GPU name, elapsed
time and peak allocated PyTorch GPU memory. Allocator memory excludes some CUDA
context/library allocations; it is not total board memory.

## Verified on 13 November 2026

- Used the existing Linux CUDA image `drone-heatmaps:model-review` with the current
  annotation source mounted read-only. This image has PyTorch 2.15.0+cu132.
- Trained the actual student segmentation model for five synthetic-data epochs on
  my NVIDIA GeForce RTX 4060, with `device: cuda` recorded in training history.
- Peak allocated PyTorch memory was 17,617,920 bytes for this small 96x96 demo.
- CUDA training/prediction/ranking checks passed, including CPU/GPU prediction
  comparison at an explicit numeric tolerance.
- Held-out synthetic GPU evaluation completed. Its score is not evidence of real
  defect accuracy, annotation-time savings or a CPU/GPU speed advantage.
- All four GPU Compose configurations parse successfully. The separate full GPU
  application images have not each been rebuilt/validated; their setup is optional.

The shared Windows development environment remains CPU-only. The above validation
used Linux GPU containers without changing that shared environment or downloading
new trained model weights. Generated data and checkpoints are excluded from Git.
