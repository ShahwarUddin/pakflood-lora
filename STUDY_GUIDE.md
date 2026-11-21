# PakFlood-LoRA Study Guide

**Project:** PakFlood-LoRA: Parameter-Efficient Adaptation of a Geospatial Foundation Model for Mapping Pakistan's Monsoon Floods and Estimating Affected Population
**Student:** M. Shahwar Uddin (CS-139/2026)
**Purpose of this guide:** explain the whole project in plain language, from the problem to the experiments running on the server, so that it can be understood, remembered and presented with confidence.

---

## 1. The project in one paragraph

Every monsoon, floods in Pakistan displace millions of people, but official flood maps are drawn by human analysts only for a few requested areas and arrive days late. We train a deep learning model that reads free Sentinel-1 radar images, which see through monsoon clouds, and colours every 10-metre pixel as dry land, permanent water or flood water. Instead of training a huge model from scratch, we take TerraMind, a foundation model already pretrained on millions of satellite images, keep it frozen, and attach tiny trainable adapters called LoRA. We test whether this cheap adaptation is as accurate as expensive full fine-tuning, whether it transfers to South Asian monsoon floods it has never seen, and whether a handful of Pakistani examples is enough to close the gap. Finally, we overlay the flood map on a population grid to count how many people live inside the flooded pixels.

---

## 2. The problem we are solving

### 2.1 Why floods in Pakistan

- The 2022 monsoon floods affected more than 33 million people, roughly one in seven Pakistanis.
- In August 2025, flash floods in Punjab and Khyber Pakhtunkhwa killed more than 300 people.
- Relief agencies need two numbers fast: where is the water, and how many people are inside it. Those numbers decide how many boats, tents and medical teams go where.

### 2.2 Why satellites, and why radar specifically

- Optical satellites (like Sentinel-2) take normal photographs. During the monsoon, clouds hide the ground for days, so optical images are often useless exactly when they are needed.
- Sentinel-1 is a radar satellite. It sends its own microwave pulses and listens for the echo. Radar passes through clouds and works at night. Smooth water reflects the pulse away from the satellite, so water looks dark in a radar image. That dark signature is what the model learns to find.
- Sentinel-1 and Sentinel-2 are free, operated by the European Copernicus programme, and revisit Pakistan every few days.

### 2.3 Why the current system is not enough

- The Copernicus Emergency Management Service (CEMS) produces expert flood maps, but only when a government requests an "activation", and only for selected areas of interest. In 2022, the two activations for Pakistan covered just four areas.
- Those maps are hand-made by analysts and take time. A trained model could instead process every new Sentinel-1 pass over the whole Indus basin automatically, within hours.

### 2.4 The machine learning task

- **Task type:** semantic segmentation. The model labels every pixel, not the whole image.
- **Classes:** 0 = background (dry land), 1 = permanent water (rivers, lakes), 2 = flood water (water that was not there before the event).
- **Input per tile:** Sentinel-1 radar before the flood and after the flood, each with two polarisation channels (VV and VH), plus a digital elevation model (DEM). We stack the "before" and "after" radar as four channels so the model can compare them directly. This is called early fusion.
- **Output:** a map of the same tile with one of the three classes per 10 m pixel.
- **Why "before and after":** permanent water is dark in both images; flood water is dark only in the "after" image. Giving the model both lets it tell the two apart.

---

## 3. The dataset: GEOID-Flood

### 3.1 What it is

- Released in August 2026 by Fondazione LINKS, Turin (Chiriaco, Barco, Bragagnolo, Rossi and Arnaudo), as an arXiv paper with data on Hugging Face and code on GitHub. Licence CC BY 4.0.
- Built from 219 real CEMS flood activations in 65 countries between 2016 and 2026. Every label was drawn by a Copernicus analyst for a real disaster and then manually checked. Nothing is synthetic.
- 12,236 tiles of 1024 x 1024 pixels at 10 m resolution (about 10 km x 10 km each), split into 7,496 train, 1,022 validation and 2,417 test tiles, plus 1,301 held-out tiles from newer events for out-of-distribution testing.
- For training, each tile is cut into 256 x 256 chips. The train split has about 278,000 paired (before/after) chips.
- Nine layers per tile: Sentinel-1 GRD and RTC (before and after), Sentinel-2 L2A, DEM, labels, cloud mask, flood mask, permanent-water mask and validity mask. We downloaded Sentinel-1 GRD, DEM and labels: 82 GB.

### 3.2 Pakistan is in the dataset

This was the biggest risk in the proposal and it is resolved. All three Pakistani activations are present:

| Activation | Event | Tiles | Official split |
|---|---|---|---|
| EMSR629 | 2022 monsoon floods | 41 | train |
| EMSR631 | 2022 monsoon floods | 5 | train |
| EMSR838 | August 2025 floods | 177 | 162 train, 15 test |

Together with three other South Asian events (EMSR205, EMSR459, EMSR851), this gives 264 South Asian tiles, about 3 percent of the training chips. The dataset catalogue has a "countries" column, which we use to build our own South Asia hold-out split.

### 3.3 Class imbalance

- On average only about 8 percent of pixels in a training chip are flood water. Most pixels are dry land.
- About 60 percent of chips contain at least some flood pixels; 40 percent contain none.
- A model that predicts "dry land" everywhere would score about 90 percent pixel accuracy and be useless. That is why we never report plain accuracy and instead use per-class IoU and F1.

---

## 4. The method

### 4.1 The foundation model: TerraMind

- A foundation model is a large neural network pretrained on a huge amount of unlabelled data so that it already understands the structure of that data. GPT models are foundation models for text; TerraMind (IBM and ESA, 2025) is one for Earth observation.
- TerraMind was pretrained on nine modalities, including Sentinel-1 radar and Sentinel-2 optical imagery, so it already knows what radar backscatter over water, cities, farmland and mountains looks like.
- The version we use is TerraMind v1 base: a Vision Transformer with 85 million parameters in the encoder. It cuts the image into patches, turns each patch into a token and processes the tokens through 12 transformer blocks with self-attention.
- The encoder produces feature maps. A decoder then turns those features back into a pixel map. The authors' configuration uses a UNet-style decoder with channel sizes 512, 256, 128 and 64 and a three-class output head. We keep the authors' decoder so our baseline numbers are directly comparable with theirs. The proposal mentioned UPerNet; this is a minor change of decoder and does not affect the hypotheses.

### 4.2 Three ways to adapt the model

1. **Frozen backbone (baseline).** The 85 M encoder weights are never updated. Only the decoder trains: 15.5 million trainable parameters. Cheapest, but the encoder cannot specialise to floods.
2. **Full fine-tuning.** Every weight trains: 101 million parameters. Most accurate in principle, but it needs the most GPU memory and time, and it risks overwriting the pretrained knowledge.
3. **LoRA (our proposal).** The encoder stays frozen, but inside each attention layer we add two small matrices, A and B, next to the query and value projections. The original weight W is 768 x 768. LoRA learns a 768 x 16 matrix and a 16 x 768 matrix, so the update is low-rank (rank r = 16). Only those small matrices and the decoder train. In our setup LoRA adds just 0.6 million parameters on top of the decoder: 16.1 million trainable in total, against 101 million for full fine-tuning.

**Why LoRA should work.** A pretrained encoder already produces good features; adapting to a new task usually needs only a small, low-rank correction, not a rewrite of every weight. LoRA was invented for language models (Hu et al., 2022) and is now standard there. Applying it to a geospatial foundation model for flood mapping, and measuring the accuracy-versus-cost trade-off, is the first contribution of this project.

### 4.3 Handling class imbalance

- **Class-weighted cross-entropy:** flood pixels count more in the loss than background pixels, so the model is punished harder for missing them.
- **Dice loss on the flood class:** Dice directly measures overlap between predicted and true flood areas, which pushes the model to get flood boundaries right rather than just getting most pixels right.
- **Stratified sampling:** chips with no flood at all are sampled less often, so each batch contains enough flood examples.

### 4.4 Training setup

- Framework: PyTorch 2.7 with TerraTorch 1.1 (IBM's fine-tuning toolkit for geospatial foundation models) on PyTorch Lightning, plus Hugging Face PEFT for LoRA.
- Mixed precision (bfloat16), AdamW optimiser, cosine learning-rate schedule, early stopping on validation flood IoU.
- Batch size 32 chips of 256 x 256 pixels; 10 epochs per run; 8,682 iterations per epoch.
- Hardware: one NVIDIA RTX PRO 6000 Blackwell (96 GB) on a remote server. Training uses only about 3 GB of GPU memory.
- Measured speed: frozen baseline 12 iterations per second (about 12 minutes per epoch), LoRA about 8 per second, full fine-tuning slower still.

---

## 5. The four hypotheses and how each is tested

| Hypothesis | Claim | Experiment | Numbers we report |
|---|---|---|---|
| H1: cheap adaptation works | LoRA gets within 1 to 2 flood-IoU points of full fine-tuning while training a small fraction of the weights | Train frozen, LoRA and full fine-tune on the same data and epochs | Flood IoU, F1, mIoU, trainable parameters, GPU memory, training time |
| H2: imbalance handling helps | Weighted CE + Dice beats the authors' plain CE on flood IoU | Same LoRA model, two losses | Flood IoU with each loss |
| H3: the South Asia gap can be closed cheaply | A model trained with no South Asian events loses accuracy on them; LoRA tuning on 10 percent of South Asian tiles recovers most of the gap | Build a South Asia hold-out split (six events moved to test), train LoRA, test; then add 10 percent of South Asian tiles to train and retrain | Flood IoU on South Asian tiles before and after few-shot adaptation |
| H4: the maps give usable impact numbers | Affected population from our map is close to the count from the official CEMS outline | Overlay predicted flood pixels on WorldPop population grid for Pakistani areas | Percentage difference in affected population |

A negative result on any hypothesis is still a result. If LoRA falls well behind full fine-tuning, that is a useful finding about geospatial foundation models.

---

## 6. Evaluation metrics explained

- **IoU (Intersection over Union)** for one class: the overlap between predicted and true pixels of that class divided by their union. 1.0 is perfect; a model that predicts nothing scores 0. We care most about flood IoU (class 2).
- **F1 score:** the harmonic mean of precision (of the pixels we called flood, how many really are) and recall (of the real flood pixels, how many we found). Recall matters most for disaster response: a missed flooded village is worse than a false alarm.
- **mIoU:** the mean of IoU over the three classes.
- **95 percent bootstrap confidence intervals:** we resample test events many times and recompute the metric, so we can say whether a difference of, say, 1.5 IoU points is real or noise.
- **Cost metrics for H1:** number of trainable parameters, peak GPU memory, and wall-clock training time per epoch.
- **Why not pixel accuracy:** because 90 percent of pixels are dry land, pixel accuracy hides whether the flood was found at all.

---

## 7. What was run and what came out

All runs finished on Thursday 19 November 2026. Everything below is measured, not planned.

### 7.1 Setup

- Everything lives in one folder in the user's home directory on the server: the authors' code, a Python 3.12 virtual environment, the 82 GB dataset (radar, elevation, labels), checkpoints, predictions and logs. Nothing was installed system-wide, and one GPU (an RTX PRO 6000, 96 GB) was used for all runs, one job at a time.
- Base configuration: the authors' early-fusion Sentinel-1 setup (pre + post radar stacked into four channels, three classes, UNet decoder on the TerraMind v1 base encoder, 256 x 256 chips, batch 32, bf16, AdamW, cosine schedule, 10 epochs, early stopping with patience 5 on validation flood IoU). The checkpoint with the best validation flood IoU is the one tested.
- LoRA is added through TerraTorch's built-in PEFT support: rank 16, alpha 32, dropout 0.05, on the query and value projections of all 12 attention blocks (0.6 M adapter parameters). The rank search variant uses rank 32 on every linear layer of the encoder (query, key, value, projection and the three MLP layers; 5.7 M adapter parameters).
- Testing: every model is run on full 1024 x 1024 tiles with a sliding window (256 window, 128 stride, averaged softmax) and the per-tile confusion matrices are kept. Confidence intervals are paired bootstraps with 2,000 resamples over tiles or over events. The chip-level TerraTorch test protocol gives numbers about one point lower and is reported in the appendix.

### 7.2 Runs completed

| Run | Purpose | Best val flood IoU | Epochs run | Time |
|---|---|---|---|---|
| baseline_frozen | H1, decoder only | 0.425 | 10 | 2.1 h |
| lora_r16 | H1, LoRA rank 16 on q,v | 0.448 | 10 | 3.1 h |
| full_finetune | H1, all 101 M weights | 0.496 | 10 | 3.1 h |
| lora_r32_all | H1 rank search, rank 32 on all linear layers | 0.488 | 10 | 3.5 h |
| lora_r16_cedice | H2, weighted CE + Dice | 0.426 | 7 (early stop) | 2.2 h |
| lora_r16_sa_holdout | H3, trained without the six South Asian events | 0.443 | 8 (early stop) | 2.4 h |
| lora_r16_sa_fewshot | H3, hold-out model plus 26 South Asian tiles | 0.450 | 7 (early stop) | 3 min |

### 7.3 H1: accuracy versus cost (official test split, 2,417 tiles from 80 events)

| Model | Flood IoU | 95% CI (tiles) | Precision | Recall | Encoder params trained | Peak GPU memory | Speed |
|---|---|---|---|---|---|---|---|
| Frozen encoder | 0.433 | 0.40 to 0.47 | 0.75 | 0.51 | 0 | 1.9 GiB | 10.6 it/s |
| LoRA r16 (q,v) | 0.493 | 0.46 to 0.52 | 0.78 | 0.57 | 0.6 M | 6.3 GiB | 7.5 it/s |
| LoRA r32 (all linear) | 0.525 | 0.49 to 0.56 | 0.77 | 0.62 | 5.7 M | 8.0 GiB | 6.5 it/s |
| Full fine-tune | 0.525 | 0.49 to 0.56 | 0.75 | 0.64 | 101 M | 7.9 GiB | 7.2 it/s |
| LoRA r16 + weighted CE + Dice (H2) | 0.500 | 0.47 to 0.53 | 0.68 | 0.65 | 0.6 M | 6.3 GiB | 7.5 it/s |

Paired differences: LoRA r16 beats the frozen encoder by 6.0 points (CI 4.4 to 7.7, clearly significant). LoRA r16 is 3.2 points behind full fine-tuning (tile-level CI 1.0 to 5.2, significant). LoRA r32 on all linear layers is level with full fine-tuning: difference +0.03 points, CI minus 1.4 to plus 1.6 at tile level, minus 2.0 to plus 2.1 at event level. On the 15 Pakistani test tiles it is even slightly ahead (0.786 versus 0.777, CI of the difference +0.1 to +2.6).

Verdict: supported, with a condition. Rank 16 on q,v only does not reach the promised one-to-two-point gap; rank 32 on all linear layers matches full fine-tuning exactly while training 5.7 M encoder parameters instead of 101 M (18 times fewer). The honest cost finding is that LoRA saves little GPU memory (about 20 percent) and no time at this batch size, because activations dominate; what it saves is what must be trained and stored per task, which is what matters for keeping one adapter per region or season.

### 7.4 H2: weighted CE + Dice (negative result)

Validation flood IoU fell from 0.448 (plain CE, the authors' loss) to 0.426 with class weights 0.3/1.0/2.0 plus a Dice term on the flood class, and training stopped early. On the official test split the two are statistically tied (0.500 versus 0.493, CI of the difference minus 0.4 to plus 1.9 points), but the trade is visible: recall rose from 0.57 to 0.65 while precision fell from 0.78 to 0.68. On Pakistan it is clearly worse (0.711 versus 0.749 on the 15 test tiles) and it over-predicts flooded area by 18 percent, which inflates the affected population to 1,025,389 (+34 percent against the official count). Say this as a finding: the weighted loss buys recall with false alarms, and for impact estimation false alarms in populated areas are expensive. Plain CE is the better choice here.

### 7.5 H3: transfer to South Asia (the strongest result)

The hold-out model was trained on 7,252 tiles with all six South Asian events removed (EMSR205, 459, 629, 631, 838, 851, 264 tiles in total). The few-shot model continues from it with 26 South Asian tiles (17 from EMSR838, four from EMSR205, three from EMSR851, one each from EMSR629 and EMSR631), 8 epochs at learning rate 5e-5, three minutes of training; the final checkpoint is used because no South Asian validation tiles exist by design. Both are tested on the same 238 South Asian tiles that are in neither set.

| Test set | Hold-out (never saw South Asia) | Few-shot (+26 tiles) | Difference |
|---|---|---|---|
| South Asia, 238 tiles | 0.679 (CI 0.65 to 0.71) | 0.720 (CI 0.69 to 0.75) | +4.1 points, CI +2.7 to +6.0 |
| Pakistan official test, 15 tiles (EMSR838-2) | 0.729 | 0.747 | +1.8 points, CI +0.7 to +3.4 |

For comparison on the same 15 Pakistani test tiles, the global models that trained on 208 Pakistani tiles score 0.712 (frozen), 0.749 (LoRA r16), 0.786 (LoRA r32) and 0.777 (full fine-tune). Two sentences to remember: the hold-out model that never saw Pakistan already beats the frozen baseline that did, and 26 tiles of few-shot adaptation bring it level with a LoRA model trained on 7,500 tiles. Gains are largest where the hold-out model was weakest (EMSR838-7 from 0.65 to 0.75, EMSR205-4 from 0.31 to 0.55). Some small areas of interest stay near zero for every model (EMSR851, EMSR205-1); they are small, scattered floods that radar alone does not resolve.

One honesty point: the 26 few-shot tiles come from the same six events as the test tiles (different tiles, no pixel overlap). This is "label a few tiles of the current flood and adapt", which is exactly the operational scenario, but it is not the same as adapting to a completely new event.

### 7.6 H4: affected population (EMSR838, 177 tiles, August 2025 Punjab floods)

Flood pixels are overlaid on WorldPop 2020 (100 m, constrained). The official CEMS outline contains 766,148 people on 2,298 km2 of flood.

| Model | Affected population | vs official | Flooded area | Population-weighted mean error per AOI | Saw EMSR838 in training? |
|---|---|---|---|---|---|
| Hold-out | 795,280 | +3.8% | 2,307 km2 (+0.4%) | 15% | no |
| Few-shot | 938,414 | +22.5% | 2,540 km2 (+10.6%) | 26% | 17 of 177 tiles |
| Frozen | 912,877 | +19.2% | 2,447 km2 | 25% | 162 of 177 tiles |
| LoRA r16 | 844,274 | +10.2% | 2,384 km2 | 14% | 162 of 177 tiles |
| LoRA r32 (all linear) | 824,004 | +7.6% | 2,410 km2 | 10% | 162 of 177 tiles |
| Full fine-tune | 824,678 | +7.6% | 2,420 km2 | 11% | 162 of 177 tiles |
| LoRA r16 + CE + Dice | 1,025,389 | +33.8% | 2,722 km2 | 35% | 162 of 177 tiles |

Verdict: supported for the honest model (hold-out, within 4 percent) but with a clear caveat. Per area of interest the hold-out errors range from minus 36 percent to plus 85 percent and partly cancel in the total, so the per-AOI error (15 percent weighted) is the number to quote alongside the total. Note also that the few-shot model has a better flood IoU (0.745 versus 0.688 on these tiles) but a worse population total: it over-predicts flood in the most populated area (EMSR838-1, +19 percent). Pixel accuracy and impact accuracy are not the same thing, because errors in populated places weigh more.

### 7.7 Figures produced

- fig_val_curves.png: validation flood IoU per epoch for the five global runs.
- fig_h1_accuracy_cost.png: official-test IoU with CIs, parameters stored per task, GPU memory and speed.
- fig_h3_transfer.png: South Asia before/after few-shot, Pakistan official test for five models, per-AOI gains.
- fig_h4_population.png: affected population per model against the official count, and per AOI.
- fig_tiles_emsr838.png: four Pakistani test tiles: radar before/after, official label, hold-out, few-shot and full fine-tune maps.
- fig_emsr838_overview.png: the whole EMSR838 activation as a mosaic over population density, official outline versus model maps.

---

## 8. What was scoped out, and how to say it

Six weeks of planned work were compressed into five days, so some items were deliberately deferred. Say this plainly; it shows judgement rather than weakness.

- **Sentinel-2 optical fusion.** The optical layer is another 142 GB and roughly doubles training time per run. We focus on radar plus elevation, which is the operationally important input because radar works through clouds. Optical fusion is listed as future work.
- **Optuna hyperparameter search.** Replaced by a two-point rank comparison: rank 16 on q,v versus rank 32 on all linear layers. The second closes most of the gap to full fine-tuning, which is the answer the search was meant to give.
- **GeoFlood-275 and the 2025 Copernicus case study.** These were stretch goals in the proposal and remain future work.
- **UPerNet decoder.** Replaced by the authors' UNet decoder to keep the comparison with the published baseline fair.

---

## 9. How to present it in ten minutes

1. **The human problem (1 min).** 33 million affected in 2022; relief needs "where" and "how many", fast; CEMS mapped only a handful of areas.
2. **Why radar and why automatic (1 min).** Clouds, night, a map per satellite pass instead of per request.
3. **The task and the data (1.5 min).** Three classes per pixel from before/after radar; GEOID-Flood with 219 events and 12,236 tiles; Pakistan's 2022 floods (EMSR629, 631) and August 2025 floods (EMSR838) are in it.
4. **The method (1.5 min).** Pretrained TerraMind encoder; three adaptation regimes side by side (frozen, LoRA, full); one slide on what LoRA is.
5. **Results (4 min).** H1 bars: frozen 0.433, LoRA r16 0.493, LoRA r32 0.525, full fine-tune 0.525; the rank-32 adapter matches full fine-tuning with 5.7 M instead of 101 M trained parameters. H3 bars: 0.679 to 0.720 with 26 tiles, and the Pakistan comparison where the hold-out model beats the frozen model that saw Pakistan. The tile figure and the Pakistan mosaic. H4: hold-out within 4 percent of the official 766 k, with the per-AOI caveat. H2 in one line: weighted loss did not help.
6. **Limits and next steps (1 min).** Deferred items; the few-shot tiles share events with the test; population grid is a model, not a census; what NDMA would need to run this each monsoon.

Keep every number on a slide, not in your head. Show the real maps; they are worth more than any table.

---

## 10. Questions the teacher may ask

**Why a foundation model instead of a plain U-Net?**
A U-Net trained from scratch only knows what it sees in the training set. TerraMind has already seen millions of radar and optical images, so it transfers better to unseen regions and needs fewer labels. The literature shows the gain is modest but consistent, and our LoRA experiment tests whether that pretrained knowledge is enough on its own.

**Why LoRA instead of just freezing or fully fine-tuning?**
Freezing is cheap but cannot specialise: 0.433 flood IoU. Full fine-tuning specialises, 0.525, but every task needs its own 101 M-parameter copy. LoRA gives 0.493 with 0.6 M parameters on q,v, and 0.525, the same as full fine-tuning, at rank 32 on all linear layers with 5.7 M. The encoder stays intact, and an agency could keep one adapter per region or season.

**LoRA r16 was 3 points behind full fine-tuning. Does that not reject H1?**
For that adapter size, yes, and we say so. Rank 32 on all linear layers closes the gap completely (0.525 versus 0.525, CI of the difference minus 1.4 to plus 1.6 points). The useful finding is where the saving really is: not GPU memory or time at this batch size, but what has to be trained and stored per task.

**Why did the weighted CE + Dice loss not help?**
It trades precision for recall (recall 0.57 to 0.65, precision 0.78 to 0.68), and IoU punishes false alarms as much as misses, so the test IoU is a statistical tie. For impact estimation the false alarms are expensive: it over-predicts flooded area by 18 percent and the affected population by 34 percent. A negative result, reported as such.

**Why do you need both the "before" and "after" radar images?**
Permanent water is dark in both; flood water is dark only after. Without the "before" image the model cannot separate a lake from a flood.

**How do you know your South Asia test is fair?**
We move whole events, not random chips, into the test set, so no pixels from the same flood appear in both train and test. Splits in the original dataset are also assigned per area of interest for the same reason.

**Is 10 percent of South Asian tiles really "few-shot"?**
It is 26 tiles of 10 km x 10 km, a few hours of analyst labelling, and three minutes of GPU time. It raised South Asian flood IoU from 0.679 to 0.720 and brought the model level with one trained on 7,500 tiles including 208 Pakistani ones. The tiles come from the same events as the test tiles (no pixel overlap), which is the operational scenario of labelling part of a current flood.

**The hold-out model has the lowest IoU on Pakistan but the best population total. Why?**
Because impact error is weighted by where people live. The few-shot model finds more flood pixels overall but over-predicts in the most populated area (EMSR838-1), and the hold-out model's per-area errors partly cancel. We therefore report the per-AOI error (15 percent for hold-out) next to the total (+3.8 percent).

**What is your evaluation metric and why not accuracy?**
Flood IoU and F1. Accuracy is dominated by dry land and would reward a model that never predicts flood.

**How reliable is the population estimate?**
WorldPop is a 100 m model-based population grid, not a census. We report the percentage difference against the count from the official CEMS outline computed on the same grid, so the error of the flood map and the error of the population grid are separated. The official outline gives 766,148 people; our best honest estimate is 795,280.

**What are the known failure modes?**
Radar confuses any smooth surface with water: airport runways, wet roads, some crops, and flat urban areas. It also over-predicts near permanent water. We report errors in towns separately for that reason.

**What would it take to use this operationally?**
A pipeline that downloads each new Sentinel-1 pass over the Indus basin, runs the model and publishes a map plus affected-population count per district. The model is small enough to run on one GPU.

**Where did the Sentinel-2 part go?**
Deferred for compute and time. Radar is the operational input during monsoon anyway, because optical imagery is cloud-covered exactly when floods happen.

---

## 11. Glossary

- **Sentinel-1:** European radar satellite; sees through clouds and at night.
- **Sentinel-2:** European optical satellite; normal multispectral images, blocked by cloud.
- **VV and VH:** the two radar polarisations. VV sends and receives vertical; VH sends vertical and receives horizontal. Water looks dark in both; VH is more sensitive to vegetation and roughness.
- **dB (decibel):** logarithmic scale for radar backscatter; the loader converts raw values to dB before normalising.
- **DEM:** digital elevation model, height of the ground per pixel. Water collects in low places, so elevation helps.
- **GRD and RTC:** two processing levels of Sentinel-1. GRD is ground-range detected; RTC is radiometrically terrain corrected. We use GRD.
- **CEMS activation (EMSRxxx):** a Copernicus Emergency Management Service rapid-mapping request for a specific disaster, numbered EMSR plus three digits.
- **Semantic segmentation:** classifying every pixel of an image.
- **Foundation model:** large network pretrained on massive unlabelled data, then adapted to many tasks.
- **TerraMind:** IBM and ESA's multimodal Earth-observation foundation model (2025).
- **Vision Transformer (ViT):** a transformer applied to image patches as tokens.
- **Encoder / decoder:** the encoder compresses the image into features; the decoder expands features back into a pixel map.
- **UNet decoder:** a decoder with skip connections at several resolutions, good at sharp boundaries.
- **LoRA:** Low-Rank Adaptation; small trainable matrices added next to frozen weights.
- **Rank (r):** the inner dimension of the LoRA matrices; higher rank means more capacity and more parameters.
- **PEFT:** parameter-efficient fine-tuning; the family of methods LoRA belongs to, and the Hugging Face library that implements them.
- **Cross-entropy loss:** standard classification loss per pixel.
- **Dice loss:** loss based on overlap between predicted and true regions.
- **IoU:** intersection over union; the main segmentation metric.
- **mIoU:** mean IoU over classes.
- **Bootstrap confidence interval:** uncertainty estimate obtained by resampling the test events.
- **WorldPop:** open 100 m population density grids from the University of Southampton.
- **NDMA / PDMA:** Pakistan's national and provincial disaster management authorities.
- **SUPARCO:** Pakistan's space agency.
- **Early fusion:** stacking inputs as channels before the encoder, as opposed to mid or late fusion.
- **Few-shot adaptation:** adapting a model with a very small number of labelled examples.
- **TerraTorch:** IBM's toolkit for fine-tuning geospatial foundation models, built on PyTorch Lightning.
