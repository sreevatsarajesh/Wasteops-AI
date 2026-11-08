# Research protocol for v1.1

Potential hypothesis: explicit unknown-class handling and calibrated uncertainty improve safe material recognition under domain shift compared with a closed-set classifier.

Required experiments:
- Compare CLIP zero-shot, MobileNetV3 transfer learning and a strong ConvNeXt baseline.
- Evaluate macro-F1, per-class recall, open-set AUROC, expected calibration error and risk-coverage curves.
- Hold out entire collection sites or datasets where possible; never split near-duplicate frames across train and test.
- Add independently collected samples from at least three disposal environments.
- Have human annotators label material, contamination, deformation and ambiguity; measure inter-annotator agreement.
- Report confidence intervals and failure cases, not just overall accuracy.

The v1.0 implementation does not claim these experiments have been completed.
