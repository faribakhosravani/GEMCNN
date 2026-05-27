# GEMCNN

**GEMCNN** is a multimodal deep learning framework for automated gemstone origin determination (OD) and treatment detection (TD) using heterogeneous analytical data sources.

---

## Overview

GEMCNN fuses measurements from up to four distinct analytical instruments:

| Modality | Type | Description |
|---|---|---|
| UV | Spectroscopy | UV spectral measurements |
| FTIR | Spectroscopy | Fourier-transform infrared spectroscopy |
| ICP-MS | Elemental analysis | Inductively coupled plasma mass spectrometry |
| XRF | Elemental analysis | X-ray fluorescence elemental profiling |

The model outputs a probability distribution over possible origins or treatment states. A confidence-thresholding mechanism allows practitioners to control the trade-off between throughput (stones processed automatically) and accuracy.

---

## Key Features

- **Multimodal fusion** — jointly processes spectroscopic and elemental data
- **Missing modality tolerance** — any subset of the four sources can be masked at inference time; the model adapts without retraining
- **Confidence thresholding** — predictions below a configurable threshold are flagged for expert review, ensuring high accuracy on the accepted subset
- **Negligible inference latency** — suitable for high-throughput laboratory pipelines
- **Outperforms human experts** — on both OD and TD tasks across multiple data source combinations

---

## Architecture

<img width="1799" alt="GEMCNN Architecture" src="https://github.com/user-attachments/assets/d21a3f12-b377-4852-9b02-c0d297f0fdd7" />

Each data source is processed by a dedicated encoder branch. The resulting embeddings are fused and passed through a classification head that produces per-class probabilities. Missing sources are masked via learned gating, illustrated by the switch symbols above.

**Confidence thresholding** operates as follows:

- If $\max_k p_k \geq \tau$ → prediction is accepted automatically
- If $\max_k p_k < \tau$ → stone is deferred to expert analysis

The threshold $\tau$ is selected during a post-training calibration phase.

---

## Performance

### Accuracy vs. Coverage

<img width="1543" alt="Accuracy vs Coverage" src="https://github.com/user-attachments/assets/2a6124ad-5973-4732-a388-f8dae72b7abe" />

Accuracy (%) vs. stones above threshold (%) for TD using XRF (left) and ICP-MS (right). XRF and ICP-MS individually underperform relative to spectroscopic sources; combining modalities recovers performance.

### GEMCNN vs. Human Experts

<img width="819" alt="Human vs GEMCNN" src="https://github.com/user-attachments/assets/d4eb2928-96de-4abf-9b3d-9b1a9904e51d" />

Each point represents a (coverage, accuracy) operating point. Circles denote GEMCNN; crosses denote human experts. GEMCNN consistently achieves higher accuracy at equal or greater coverage across all data source combinations.

### Confusion Matrices — Treatment Detection

<img width="1530" alt="Confusion Matrices" src="https://github.com/user-attachments/assets/8a2d76ac-0496-42ee-b853-cfa6c8ee5c86" />

TD confusion matrices under three operating modes:

| Mode | Description |
|---|---|
| None | No thresholding; all stones classified |
| Mode 1 | Conservative threshold; moderate deferral rate |
| Mode 2 | Strict threshold; low deferral rate, highest accuracy |


---

## Quick Start

```python
from gemcnn import GEMCNN, fit

# Initialize model for origin determination
model = GEMCNN(
    uv_input_len=1200,
    ftir_input_len=1700,
    xrf_features=30,
    icpms_features=40,
    task='od',          # 'od' for origin determination, 'td' for treatment detection
)

# Train
fit(model, train_loader, val_loader, device, uv_mean, ftir_mean, elem_mean)
```

### Inference with Missing Modalities

GEMCNN handles incomplete inputs natively. Simply pass `None` for any unavailable source:

```python
probs = model(uv=uv_tensor, ftir=ftir_tensor, xrf=None, icpms=None)
```

### Confidence Thresholding

```python
import torch

probs = model(uv=uv_tensor, ftir=ftir_tensor, xrf=xrf_tensor, icpms=icpms_tensor)
max_prob, predicted_class = torch.max(probs, dim=-1)

tau = 0.85  # set during calibration
accepted = max_prob >= tau
```

---

## Tasks

| Task | Argument | Description |
|---|---|---|
| Origin Determination | `task='od'` | Predicts geographic origin of a gemstone |
| Treatment Detection | `task='td'` | Predicts whether a stone has undergone heat treatment |
---

## License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE) for details.
