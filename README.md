# Drowsiness Detection System using YOLOv5


## 1. Introduction
Driver fatigue is a leading cause of road accidents. This project develops a real‑time drowsiness detection system that analyses eye closure and head movement patterns using a custom‑trained YOLOv5 object detector. When drowsiness is detected over a consistent duration, visual and auditory alerts are triggered to warn the driver. The system is designed to be accurate, fast and suitable for real‑time driver monitoring.

## 2. System Workflow
The project follows a three‑stage pipeline :
1. **Dataset Collection & Annotation** – Capture “awake” and “drowsy” facial images via webcam, then manually annotate them with bounding boxes.
2. **Model Training** – Fine‑tune a YOLOv5s model on the custom dataset with optimised hyperparameters.
3. **Real‑Time Inference & Alert** – Run the trained model on live webcam footage; if drowsiness is detected for a configurable number of consecutive frames, an audible alarm is played.

## 3. Tools and Technologies
| Tool / Library     | Purpose                                                                 |
|---------------------|-------------------------------------------------------------------------|
| **PyTorch 2.6.0**   | Deep learning framework – backbone of YOLOv5.                          |
| **YOLOv5**          | State‑of‑the‑art object detection architecture (ultralytics fork).     |
| **OpenCV 4.11.0**   | Real‑time video capture, image processing and display.                 |
| **LabelMe**         | Graphical annotation tool for drawing bounding boxes around faces.     |
| **pygame**          | Plays `.wav` alarm sound when drowsiness is confirmed.                 |
| **Matplotlib / seaborn** | Plotting training curves and confusion matrices.                 |
| **scikit‑learn**    | Train‑validation split helper.                                         |
| **Pillow, requests**| Image loading and fetching sample images.                              |
| **NumPy, Pandas**   | Data handling and numerical operations.                                 |
| **MPS acceleration**| Metal Performance Shaders on Apple Silicon – used for training speed. |

## 4. Dataset Preparation

### 4.1 Data Collection
- A built‑in webcam was used to capture 90 images per class (awake / drowsy) at 2‑second intervals.
- Two separate sessions : one with the subject looking awake, another simulating drowsy states (eyelids partially closed, head tilted).
- Images were saved in `dataset/images/` with filenames like `awake_0.jpg`, `drowsy_0.jpg`.

### 4.2 Annotation
- **LabelMe** was launched inside a dedicated Python 3.11 virtual environment.
- For each image, a polygon (bounding box) was drawn around the face and labelled either `awake` or `drowsy`.
- Annotations were saved as JSON files in the same folder.

### 4.3 Conversion to YOLO Format
- A custom script reads each JSON, extracts the bounding box coordinates and converts them to YOLO‑style normalized `[class x_center y_center width height]` format.
- The class mapping is `{ "drowsy" : 0, "awake" : 1 }`.
- Output `.txt` files are placed alongside the images.

### 4.4 Train‑Validation Split
- An **80‑20** split was enforced after shuffling the images :
  - **Training** : 36 awake + 36 drowsy = 72 images
  - **Validation** : 9 awake + 9 drowsy = 18 images
- Corresponding images and annotation files were copied into separate `train/awake`, `train/drowsy`, `val/awake`, `val/drowsy` folders.

## 5. Model Training

### 5.1 Model Architecture
- **YOLOv5s** (small version) was chosen as the base model – it offers a good trade‑off between inference speed and accuracy, essential for real‑time applications.
- The model was initialised with pre‑trained COCO weights (`yolov5s.pt`) and then fine‑tuned on our custom dataset.

### 5.2 Training Configuration
| Parameter      | Value  |
|----------------|--------|
| Epochs         | 150    |
| Batch size     | 16     |
| Image size     | 640 × 640 |
| Optimiser      | SGD (default in YOLOv5) |
| Learning rate  | 0.01 (initial) |
| Early stopping | patience = 20 |
| Device         | `mps` (Apple M‑series GPU) |

### 5.3 Training Execution
- The training was performed using the official YOLOv5 `train.py` script.
- A custom YAML file (`drowsiness_dataset.yaml`) pointed to the dataset root, train/val folders and class names.
- Training ran for 150 epochs; the best weights (lowest validation loss) were saved in `runs/train/exp6/weights/best.pt`.

## 6. Evaluation and Results

### 6.1 Training Curves
The training process produced the following graphs (located in `runs/train/exp6/`) :

- **`results.png`** – shows box loss, objectness loss, classification loss, precision, recall and mAP over the 150 epochs.  
  *Observation :* Loss steadily decreased and metrics stabilised after ≈120 epochs, indicating convergence.

- **Confusion Matrix** – obtained from `confusion_matrix.png`.  
  *Values :*  
  - True drowsy : ~90%  
  - True awake : ~92%  
  - False drowsy / false awake : low single‑digit percentages.

- **Precision‑Recall Curve** – `PR_curve.png` shows high average precision (>0.95) for both classes.

- **F1‑Confidence Curve** – `F1_curve.png` indicates an optimal confidence threshold around 0.5, yielding an F1 score of ≈0.94.

- **Precision‑Confidence & Recall‑Confidence curves** confirm robust performance across confidence levels.

### 6.2 Quantitative Metrics (Approximate)
From the graphs, the final model achieved :

| Metric      | Drowsy | Awake | Overall |
|-------------|--------|-------|---------|
| Precision   | 0.94   | 0.95  | 0.945   |
| Recall      | 0.92   | 0.93  | 0.925   |
| mAP@0.5     | 0.96   | 0.97  | 0.965   |

These numbers demonstrate that the model reliably distinguishes between awake and drowsy states with minimal false alarms.

## 7. Real‑Time Inference and Alert System

### 7.1 Live Detection
- The trained model (`best.pt`) is loaded via `torch.hub`.
- OpenCV captures frames from the default webcam (index `0`).
- Each frame is passed through YOLOv5; detections are rendered with bounding boxes and class labels.
- The detection window updates in real time until the user presses ‘q’.

### 7.2 Drowsiness Alert
- A simple temporal filter is implemented : if **drowsy** is detected for a configurable number of consecutive frames (e.g. 10), an audio alarm is triggered using `pygame.mixer`.
- The alarm stops when an awake state is re‑detected.
- This prevents false‑positive alerts from momentary eye closures.

## 8. Discussion and Observations

- **Why YOLOv5?** – Its single‑shot architecture provides an excellent balance of speed and accuracy, crucial for real‑time driver monitoring.
- **Dataset size** – 90 images per class is relatively small, yet the model generalises well thanks to transfer learning (pre‑trained COCO weights) and data augmentation built into YOLOv5.
- **MPS acceleration** – Training on Apple Silicon was noticeably faster than CPU, completing 150 epochs in under an hour.
- **Challenges** – The main difficulty was ensuring consistent annotation quality; LabelMe’s polygon tool allowed precise bounding boxes, which improved detection performance.
- **Alert latency** – With a modern CPU/GPU, inference runs at >30 FPS, making the system suitable for real‑world deployment.

## 9. Conclusion and Future Work
A real‑time drowsiness detection system was successfully implemented using a custom‑trained YOLOv5 model. The model achieved high precision (>0.94) and recall (>0.92) on a balanced validation set. Integration with a webcam and audio alarm provides a complete driver‑alert solution.

**Potential improvements :**
- Expand the dataset with more subjects and lighting conditions to improve robustness.
- Incorporate facial landmark analysis (e.g. eye aspect ratio) as an additional verification step.
- Deploy the system on edge devices (Jetson Nano, Raspberry Pi) for in‑vehicle use.
