# Smart Plant Health Monitor

Detects crop disease from a single leaf photograph and forecasts nutrient
requirements from sensor history.

A ResNet50 classifier reaches **90.82% validation accuracy across 38 disease
and healthy classes**, a YOLOv5 detector localises affected leaf regions at
**86.6% mAP@0.5**, and an LSTM predicts per-nutrient need levels from 30-day
sensor sequences. All three are served behind a FastAPI service with a
Streamlit interface.

Built for DATA602 at the University of Maryland.

![Streamlit interface classifying a cedar apple rust sample and showing predicted nutrient needs](docs/ui_prediction.png)

---

## Quick start

Everything needed to serve predictions is in the repository, including the
trained weights.

```bash
git clone https://github.com/dhruvdubey3123/plant_health_monitor.git
cd plant_health_monitor

python -m venv venv
source venv/bin/activate          # Windows: .\venv\Scripts\Activate.ps1
pip install -r requirements.txt

python src/api/main.py            # API on :8000
streamlit run src/ui/app.py       # UI on :8501, separate terminal
```

Or with Docker:

```bash
docker compose -f docker/docker-compose.yml up --build
```

Verify the API directly:

```bash
curl -X POST http://localhost:8000/predict -F "file=@path/to/leaf.jpg"
```

---

## Results

| Task | Model | Metric | Result |
| --- | --- | --- | --- |
| Disease classification | ResNet50, 38-class head | Best validation accuracy | **90.82%** |
| Leaf region detection | YOLOv5, 5 classes | mAP@0.5 | **86.6%** |
| Nutrient-need forecasting | LSTM, 2 layers, 128 hidden | Validation accuracy | **69.4%** |

ResNet50 trained 25 epochs; final-epoch validation accuracy was 89.38% with the
best checkpoint at 90.82%. Per-epoch curves are in
`models/resnet/training_history.json` and `models/lstm/training_history.json`.
YOLO training curves, the confusion matrix, PR and F1 curves, and annotated
validation batches are under `models/yolo/plant_disease/`.

### Observed behaviour on held-out samples

| Input | Predicted | Confidence |
| --- | --- | --- |
| `AppleCedarRust1.JPG` | Apple / Cedar apple rust | 98.1% |
| `PotatoHealthy1.JPG` | Potato / healthy | 93.8% |
| `TomatoEarlyBlight1.JPG` | Tomato / **Late** blight | 62.0% (Early blight second at 22.5%) |

The third row is a real failure mode worth naming: early and late blight on
tomato share visual symptoms, and the classifier confuses them. Lesion margin
and halo pattern are the distinguishing features, and neither survives the
224px centre crop reliably.

---

## How it works

```
leaf image ──► YOLOv5 detection ──► region crop
                                         │
                                         ▼
                              ResNet50 classifier ──► 1 of 38 classes
                                                      + top-3 confidences

sensor history ──► LSTM ──► per-nutrient need level
(30 days × 11 features)      (low / medium / high) × 10 nutrients
```

**Classification.** ResNet50 initialised from ImageNet, final layer replaced
with dropout(0.5) and a 2048 to 38 linear head. Images resized to 256,
centre-cropped to 224, normalised with ImageNet statistics. Batch size 32,
Adam at 1e-3, early stopping with patience 10.

**Detection.** YOLOv5 trained on a 5-class subset to localise leaf and lesion
regions before classification, so the classifier sees the leaf rather than the
background.

**Forecasting.** A 2-layer LSTM over 30-day sequences of 11 sensor features
(temperature, humidity, soil moisture, light, NPK and others), with a head
emitting 3 logits per nutrient across 10 nutrients.

### Endpoints

| Method | Path | Returns |
| --- | --- | --- |
| `GET` | `/` | API metadata and model summary |
| `GET` | `/health` | Health check |
| `POST` | `/predict` | Disease class with top-3 confidences |
| `POST` | `/nutrition` | Nutrient need levels with per-class probabilities |
| `POST` | `/analyze` | Both, from a single image upload |

---

## Data

**Images.** [New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset),
roughly 87,000 RGB leaf photographs across 38 crop and condition classes
spanning apple, blueberry, cherry, corn, grape, orange, peach, pepper, potato,
raspberry, soybean, squash, strawberry and tomato.

**Sensor series.** Generated synthetically by
`src/data_preparation/generate_timeseries.py` for 1,000 simulated plants over
30-day windows. This is the honest weak point of the nutrient model: no real
field sensor data was available.

---

## What is in the repository

| Path | Contents |
| --- | --- |
| `src/data_preparation/` | Dataset exploration, ResNet and YOLO data prep, sensor generation |
| `src/models/` | `train_resnet.py`, `train_yolo.py`, `train_lstm.py` |
| `src/api/` | `main.py` (FastAPI routes), `inference.py` (model loading and prediction) |
| `src/ui/app.py` | Streamlit interface |
| `models/` | Trained weights and training histories for all three models |
| `config/config.yaml` | Paths, model settings and training hyperparameters |
| `docker/` | `Dockerfile` and `docker-compose.yml` |

Weights are committed, so the clone is roughly 130MB: ResNet 90MB, YOLO 28MB,
LSTM under 1MB. To retrain rather than use them, run
`python src/data_preparation/prepare_resnet_data.py` followed by
`python src/models/train_resnet.py`.

---

## Applying this to overhead imagery

The design is two-stage detect-then-classify, the same structure used in
aerial and satellite analysis. Moving it to overhead imagery changes the data
handling rather than the architecture:

**Tiling and georeferencing.** Satellite scenes are orders of magnitude larger
than a 224px crop, so input becomes fixed-size overlapping tiles and
predictions are stitched back to geographic coordinates. Overlap has to exceed
the largest expected object or targets get cut at tile boundaries.

**Spectral channels.** The ResNet stem expects 3-channel RGB. Multispectral or
SAR input needs a widened first convolution, with pretrained weights averaged
or repeated across the added channels rather than randomly initialised.

**Class imbalance.** The 38 leaf classes here are roughly balanced. Overhead
targets rarely are, which shifts evaluation away from plain accuracy toward
per-class recall and precision-recall area.

**Validation splits.** Random splits leak when neighbouring tiles land on both
sides. Splits have to be held out by geography or capture date, which usually
drops reported performance and makes it meaningful.

**Label scarcity.** Leaf datasets are large and public; overhead annotation is
the expensive constraint. That pushes toward pretrained backbones, weak
supervision and careful use of whatever labelled area exists.

What transfers unchanged is the scaffolding: the detect-then-classify split,
checkpointing and history logging in `src/models/`, and the inference-service
pattern in `src/api/`.

---

## Limitations

- The nutrient LSTM trains on synthetic sensor data. Its 69.4% accuracy says
  nothing about performance on real field sensors.
- `/nutrition` generates a mock sequence when none is supplied. Production use
  needs real readings wired in.
- Validation splits are random rather than stratified by field or capture
  session, so 90.82% is likely optimistic against genuinely new growing
  conditions.
- Early versus late blight on tomato is unreliable, as shown above.
- The dataset is laboratory-style photographs on uniform backgrounds. Field
  photographs with mixed foliage, shadow and varied lighting are a harder
  problem than these numbers suggest.

---

## Team

DATA602 group project, University of Maryland.

| Name | UID |
| --- | --- |
| Raj Prasad Ambavane | 121998644 |
| Dhruv Dubey | 122129887 |
| Vansh Pradeep Jain | 122082892 |
| Shantanu Ramavat | 121997708 |

This repository holds the model training, inference and service work for this
copy of the project. The group's shared repository is
[vjain0812/DATA602_smartPlant_health](https://github.com/vjain0812/DATA602_smartPlant_health).

---

## License

Academic coursework. The dataset carries its own Kaggle licence.
