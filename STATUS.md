# Verification and remaining work

Reliability update (12 November 2026): approved-mask hashes are verified on import,
approval resets on changed review inputs, checkpoint identity is preserved, binary
mask values are checked, and optional acquisition groups cannot cross splits.
Legacy audits require re-review. Regression checks include changed masks and actual
Streamlit checkbox reset behavior for changed images/checkpoints. Earlier results
below remain synthetic; real annotation savings and GPU behavior are not established.

Working: small PyTorch segmentation model; seeded training; validation-selected weights;
held-out test evaluation; four pool-selection modes; predicted-mask review UI; correction
upload; mask/provenance/time records; explicit import to a new training round.

Verified: five training epochs on synthetic red-circle images; held-out test IoU ~0.959;
four pool samples ranked; eight unit checks passed. These toy results do not establish
real-data performance or annotation savings. Review/import and UI checks are recorded
in the development chat; a real person must still review labels for real experiments.

Pending: SAM teacher, interactive mask painting (currently upload a corrected PNG),
comparative multi-round experiments on real images, and GPU profiling. The local
verification environment is CPU-only. No real images, reviewed labels, weights, or
private paths are included in the source repository.
