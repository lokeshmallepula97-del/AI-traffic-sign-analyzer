# Technical Methodology: Indian Traffic Sign Recognition using BanTrafficNet

---

## 1. Introduction & Theoretical Foundation
Traffic Sign Recognition (TSR) is an essential component of Advanced Driver Assistance Systems (ADAS) and autonomous vehicle navigation. The primary technical bottleneck in subcontinental road environments is the scarcity of standardized datasets and the computational heaviness of standard deep vision models (e.g., VGG19, ResNet50).

This project adapts the lightweight **BanTrafficNet** architecture (*Scientific Reports / Nature Portfolio*, 2026) to the **Indian Road Congress (IRC:67)** classification standards.

---

## 2. Dataset Pipeline & Ingestion Architecture

```
[Raw Indian Roadside Imagery] -> (3,277 Original Images, 56 Classes)
                │
                ▼
[Integrity Verification Gate]
  ├─ PIL Image decode & EXIF orientation normalization
  ├─ SHA-256 Hash Deduplication (0 Collisions)
  └─ IRC:67 Class Taxonomy Mapping (56 Sign Types)
                │
                ▼
[Stratified Partitioning (Prior to Augmentation)]
  ├─ Training Split (75.0%): 2,459 images
  ├─ Validation Split (12.5%): 409 images
  └─ Held-Out Test Split (12.5%): 409 images
                │
                ▼
[Training-Only Data Augmentation]
  ├─ Random Rotation: +-15 degrees
  ├─ Random Contrast & Brightness: +-10%
  ├─ Random Zoom & Translation: +-10%
  └─ Blur & Grayscale Invariance
```

---

## 3. BanTrafficNet Architecture Specifications

BanTrafficNet uses **valid 2D convolutions** across three distinct hierarchical blocks, progressively expanding channel capacity from 32 to 256:

$$\text{Spatial Output Resolution} = \left\lfloor \frac{W - K + 2P}{S} \right\rfloor + 1$$

With kernel $K=3$, padding $P=0$, and stride $S=1$:
1. **Input:** $96 \times 96 \times 3$
2. **Conv Block 1:**
   - $\text{Conv2D}(32, 3 \times 3, \text{valid}) \rightarrow 94 \times 94 \times 32$
   - $\text{Conv2D}(32, 3 \times 3, \text{valid}) \rightarrow 92 \times 92 \times 32$
   - $\text{MaxPool2D}(2 \times 2, \text{stride}=2) \rightarrow 46 \times 46 \times 32$
3. **Conv Block 2:**
   - $\text{Conv2D}(64, 3 \times 3, \text{valid}) \rightarrow 44 \times 44 \times 64$
   - $\text{Conv2D}(64, 3 \times 3, \text{valid}) \rightarrow 42 \times 42 \times 64$
   - $\text{MaxPool2D}(2 \times 2, \text{stride}=2) \rightarrow 21 \times 21 \times 64$
   - $\text{Dropout}(0.25)$
4. **Conv Block 3:**
   - $\text{Conv2D}(128, 3 \times 3, \text{valid}) \rightarrow 19 \times 19 \times 128$
   - $\text{Conv2D}(256, 3 \times 3, \text{valid}) \rightarrow 17 \times 17 \times 256$
   - $\text{MaxPool2D}(2 \times 2, \text{stride}=2) \rightarrow 8 \times 8 \times 256$
   - $\text{Dropout}(0.25)$
5. **Classification Block:**
   - $\text{Flatten} \rightarrow 8 \times 8 \times 256 = 16,384 \text{ features}$
   - $\text{Dense}(256, \text{ReLU}) \rightarrow 256$
   - $\text{Dropout}(0.50)$
   - $\text{Dense}(C=56, \text{Softmax}) \rightarrow 56$

### Exact Parameter Formulation
$$\text{Total Parameters} = 4,629,152 + (257 \times C)$$
For $C = 56$ classes:
$$\text{Total Parameters} = 4,629,152 + 14,392 = \mathbf{4,643,544}$$

---

## 4. Training & Optimization Methodology

* **Loss Function:** Sparse Categorical Cross-Entropy
  $$\mathcal{L} = -\sum_{i=1}^C y_i \log(\hat{y}_i)$$
* **Optimizer:** Adam ($\text{lr} = 0.001, \beta_1 = 0.9, \beta_2 = 0.999$)
* **Learning Rate Scheduling:** `ReduceLROnPlateau` reducing LR by $0.5$ if validation loss stagnates for 5 epochs.
* **Regularization:** Checkpointing highest validation accuracy weights with `EarlyStopping` (patience = 15).
* **Target Inference Speed:** $<10$ ms per frame on embedded edge devices ($>100$ FPS).

---

## 5. Indian Road Congress (IRC:67) Integration

Each recognized sign is mapped to its regulatory meaning under Indian law:
1. **Mandatory / Regulatory:** Enforceable under the Motor Vehicles Act; violators face legal penalties.
2. **Cautionary / Warning:** Warns motorists of upcoming hazards (curves, pedestrians, speed breakers).
3. **Informatory:** Directs drivers to essential facilities (hospitals, fuel stations, parking).
