# Edge AI Course – Project Report

# Real Time Helmet Detection Project
This project was completed as part of the *Edge AI Course*, which focuses on real-time AI deployment on edge devices.

🔗 Course Website: https://www.samy101.com/edge-ai-26/

---

## 1. Problem Statement, Motivation & Objectives
Road accidents involving two-wheeler riders without helmets are a major safety concern. Manual monitoring by traffic authorities is inefficient and not scalable. This project addresses the need for an automated system that can detect helmet usage in real time using computer vision.

The motivation behind this work is to build a low-cost, real-time edge AI system that can operate without reliance on cloud infrastructure. By performing inference on edge devices like Raspberry Pi, the system reduces latency, improves privacy, and minimizes bandwidth usage.

### Objectives:

- Detect riders, helmet status (with/without) in real time using a YOLOv5-based pipeline
- Deploy the trained model on a Raspberry Pi 5 using ONNX Runtime, with no PyTorch dependency on the device
- Optimize the system for real-time performance by minimizing latency and maximizing stable FPS for efficient and timely detection on edge devices. 
- Implement a TCP client-server architecture for distributed frame capture and inference
- Evaluate model quality using Precision, Recall, mAP@0.5, and mAP@0.5:0.95
- Achieve measurable on-device inference with target latency under 1 second per frame on CPU
---

## 2. Proposed Solution

The system implements a two-node distributed inference pipeline, where the laptop acts as a client for real-time frame capture, compression, and TCP streaming, while the Raspberry Pi 5 serves as the inference server running the ONNX model and returning structured JSON detections, which are rendered as bounding boxes on the client for visualization.


**System Pipeline:**

```
Webcam (Laptop)
  → JPEG Compression (quality 80)
    → TCP Socket (length-prefixed framing)
      → Raspberry Pi 5: ONNX Runtime Inference
        → JSON Detection Results
          → TCP Socket → Laptop (Bounding Box Rendering)
```

**Design Justifications:**

| Component | Choice | Rationale |
|---|---|---|
| Detection model | YOLOv5s | Lightweight single-stage detector; strong accuracy-to-compute ratio for CPU inference |
| Inference runtime | ONNX Runtime | Removes PyTorch from the Pi; delivers CPU-optimized execution with lower memory footprint |
| Transport | TCP + length-prefixed framing | Guarantees ordered, lossless delivery of variable-length image payloads; prevents partial reads |
| Compression | JPEG at quality 80 | Reduces per-frame TCP payload size with negligible detection quality impact |

---

## 3. Hardware & Software Setup

**Hardware:**

| Component | Specification |
|---|---|
| Edge Inference Device | Raspberry Pi 5 (8GB RAM) |
| Client Device | Laptop (frame capture and result visualization) |
| Camera | USB Webcam |

The Raspberry Pi 5 was selected for its significantly improved CPU performance over prior Pi generations, making CPU-bound ONNX inference feasible for near-real-time use without a dedicated hardware accelerator.

**Software:**

| Component | Tool / Version |
|---|---|
| OS (Raspberry Pi) | Raspberry Pi OS 64-bit (Bookworm) |
| Language | Python 3.11 |
| Detection Framework | YOLOv5 (Ultralytics) |
| Inference Runtime | ONNX Runtime (CPU) |
| Vision Library | OpenCV |
| Communication | Custom TCP protocol (length-prefixed binary framing) |
| Utilities | NumPy, JSON |

---

## 4. Data Collection & Dataset Preparation

**Source:** link : https://www.kaggle.com/datasets/aneesarom/rider-with-helmet-without-helmet-number-plate
Public Kaggle helmet detection dataset with YOLO-format annotations (normalized bounding boxes).

**Class Distribution:**

| Class | Approx. Instances |
|---|---|
| Rider | ~120 |
| Number Plate | ~116 |
| Without Helmet | ~93 |
| With Helmet | ~64 |

The dataset is fairly well-structured, covering all key classes required for the task. While the “With Helmet” class has slightly fewer samples compared to others, it still provides useful learning signals for the model. With further data collection, especially for this class, performance can be improved even more.

Although the overall dataset size is relatively small, it is sufficient to demonstrate the working pipeline and achieve meaningful results. Expanding the dataset in the future will help the model generalize better across different environments, lighting conditions, and real-world scenarios.

**Data Split:** Standard train / validation / test partition.

**Augmentation pipeline:**
- **Horizontal flip** - helps the model handle different riding directions and improves generalization
- **HSV color jitter** - makes the model more robust to changes in lighting and camera exposure.
- **Mosaic augmentation** (YOLO default) - combines four images into one during training, which improves detection of small and partially visible objects like helmets and number plates.



**Key Dataset Characteristics & Challenges:**

- **Occlusion:** Helmets and number plates are sometimes partially hidden in real traffic scenes, helping the model learn to handle such cases.  
- **Class distribution:** The *"With Helmet"* class has fewer samples, providing an opportunity to further improve performance with additional data.  
- **Lighting variability:** The dataset includes both bright and low-light conditions, making the model more adaptable to real-world environments.  
---

## 5. Model Design, Training & Evaluation

### Architecture: YOLOv5s

| Component | Details |
|---|---|
| Backbone | CSPDarknet - performs efficient feature extraction using cross-stage partial connections, reducing computational cost while maintaining strong feature representation |
| Neck | PANet - enhances feature fusion by combining low-level and high-level features, enabling robust detection across different object scales |
| Detection Heads | Three detection scales - specialized heads for small, medium, and large objects, improving detection accuracy for both tiny objects (e.g., helmets) and larger ones (e.g., riders) |

YOLOv5s is a lightweight, single-stage object detection architecture designed for high-speed inference with competitive accuracy. The backbone (CSPDarknet) extracts rich spatial and semantic features from input images while keeping computation efficient, which is important for edge deployment.

The neck (PANet) plays a key role in aggregating features from multiple layers, allowing the model to retain both fine-grained details and high-level context. This is particularly useful in scenarios where objects vary significantly in size, such as small number plates and larger rider regions within the same frame.

The detection heads operate at three different scales, ensuring that objects of varying sizes are effectively detected. This multi-scale detection capability is critical for real-world traffic scenes, where object sizes can change due to distance, camera angle, and perspective.

Overall, YOLOv5s provides a strong balance between speed and accuracy, making it well-suited for real-time inference on resource-constrained devices like the Raspberry Pi 5.

**Training Configuration:**

| Parameter | Value |
|---|---|
| Input resolution | 640 × 640 |
| Batch size | 16 |
| Epochs | 50 |
| Optimizer | SGD |
| Initial learning rate | 0.01 |
| LR schedule | Cosine annealing with linear warm-up |

**Evaluation Results:**

| Metric | Value |
|---|---|
| Precision | 0.9527 |
| Recall | 0.8891 |
| mAP@0.5 | 0.9430 |
| mAP@0.5:0.95 | 0.6956 |

**Per-Class AP@0.5:**

| Class | AP@0.5 |
|---|---|
| With Helmet | 0.972 |
| Without Helmet | 0.914 |
| Rider | 0.929 |
| Number Plate | 0.956 |

**Analysis:**
- High precision (0.9527) confirms a very low false positive rate, the model fires reliably when a detection is warranted
- Recall of 0.8891 reflects a small miss rate, most likely attributable to occlusion and the relatively limited "with helmet" training samples
- Strong mAP@0.5 (0.9430) demonstrates robust multi-class detection performance across all four classes at standard IoU
- The drop to mAP@0.5:0.95 (0.6956) is expected for a lightweight model like YOLOv5s and reflects reduced localization precision at stricter IoU thresholds - a known speed-accuracy trade-off
- All three loss components (box, objectness, classification) decreased consistently across 50 epochs on both train and validation sets with no signs of overfitting, confirming good generalization given the dataset size
- The optimal F1 score across all classes is **0.92 at a confidence threshold of 0.656**
- The confusion matrix shows "without helmet" recall at 0.80, with 0.20 of instances misclassified as "with helmet" - the most safety-critical error mode, driven by occlusion and class imbalance
### Figures

**Figure 1: Training performance over 50 epochs showing loss curves (box, objectness, classification) along with precision, recall, and mAP metrics. The steady decrease in losses and consistent validation trends indicate stable training and good generalization.**

<img width="2400" height="1200" alt="fig_results" src="https://github.com/user-attachments/assets/e434bc7f-ec39-4362-90f7-fd7e9976161a" />

---

**Figure 2: Confusion matrix on the validation set. The model shows strong performance across all classes, with high recall for most categories. Minor confusion is observed between "with helmet" and "without helmet", primarily due to occlusion and visual similarity in challenging cases.**

<img width="3000" height="2250" alt="fig_confusion_matrix" src="https://github.com/user-attachments/assets/10e63ab4-1136-4470-a251-75b0ef8cbcdb" />

---

**Figure 3: Precision–Recall curves for each class and overall model performance (mAP@0.5 = 0.943). The curves demonstrate high precision across a wide recall range, indicating a well-calibrated and reliable detector.**

<img width="2250" height="1500" alt="fig_pr_curve" src="https://github.com/user-attachments/assets/93204153-249f-40a5-9524-216b514c6cb2" />

---
**Figure 4: Training loss curves (box, objectness, and classification) over epochs. All losses decrease steadily, indicating stable training and good model convergence without signs of overfitting.**
  
<img width="865" height="535" alt="image" src="https://github.com/user-attachments/assets/bc28c248-39b1-4917-920b-df491d493f8d" />
---
**Figure 5: Validation loss curves (box, objectness, and classification) over epochs. The consistent decrease with minor fluctuations indicates stable generalization and no significant overfitting.**

<img width="865" height="535" alt="image" src="https://github.com/user-attachments/assets/dac2072d-8bab-4e9c-a31c-d7f5f43b5b29" />
---
**Figure 6: Precision and recall trends over epochs. Both metrics improve steadily, indicating better detection performance and balanced learning as training progresses.**
<img width="865" height="535" alt="image" src="https://github.com/user-attachments/assets/73195d8f-dd4a-4afa-bc8c-5157e3d7e075" />

---
**Figure 7: Mean Average Precision (mAP@0.5 and mAP@0.5:0.95) over epochs. The consistent increase reflects improved detection accuracy and localization performance across training.**
<img width="865" height="535" alt="image" src="https://github.com/user-attachments/assets/96fdf43e-dc69-4b40-916b-3cc094df39f5" />

---
**Figure 8: F1–confidence curves for all classes. The model achieves an optimal F1 score of approximately 0.92 at a confidence threshold around 0.65, indicating a good balance between precision and recall.**
<img width="2048" height="1365" alt="image" src="https://github.com/user-attachments/assets/b52b8733-cd7d-4ff5-a7da-902c8ec47013" />

---
**Figure 9: Precision–confidence curves for all classes. Precision increases with higher confidence thresholds, indicating reduced false positives at stricter detection confidence levels.**
<img width="2048" height="1365" alt="image" src="https://github.com/user-attachments/assets/e39ec3f2-2763-43c5-980a-974e9d26c376" />

---
**Figure 10: Recall–confidence curves for all classes. Recall decreases as confidence increases, highlighting the trade-off between capturing all detections and maintaining prediction certainty.**
<img width="2048" height="1365" alt="image" src="https://github.com/user-attachments/assets/06f2ec16-8b56-4237-bc86-5337d4c46604" />

---

## 6. Model Conversion & Efficiency Metrics

**Conversion:** PyTorch (`.pt`) → ONNX (`.onnx`) via `torch.onnx.export`

ONNX export is a **runtime optimization**, not a compression technique. It serializes the trained computation graph into a portable, framework-independent format executable by ONNX Runtime - a lightweight inference engine with CPU-specific kernel optimizations unavailable in standard PyTorch. This removes the need to install PyTorch on the Raspberry Pi, reducing the deployment environment's size, startup time, and memory footprint significantly.

**On-Device Performance (Raspberry Pi 5, CPU-only):**

| Metric | Value |
|---|---|
| Inference latency | ~969 ms / frame |
| Throughput | ~1.2 FPS |
| Inference runtime | ONNX Runtime (CPU) |
| PyTorch required on device | No |

**Trade-offs and Bottlenecks:**
- At ~969 ms per frame (~1.2 FPS), the system is suitable for controlled-intersection violation flagging, but insufficient for high-speed or multi-lane video streams
- The primary bottleneck is CPU-only execution; a dedicated NPU such as the Coral TPU or Hailo-8 could reduce latency by 10–20× while preserving detection accuracy
- JPEG compression at quality 80 effectively reduces TCP transmission overhead with negligible impact on detection quality at typical traffic camera distances
- ONNX Runtime removes PyTorch overhead on the Pi, lowering RAM usage and improving startup time — but does not address the fundamental compute limitations of a CPU-only deployment
- For production use, INT8 quantization combined with hardware acceleration is the natural next step toward achieving real-time throughput (≥10 FPS)

## 7. Model Deployment & On-Device Performance

### Deployment Steps
The deployment pipeline is designed to enable efficient edge inference on the Raspberry Pi 5 using a lightweight ONNX-based runtime. The complete flow, including preprocessing and postprocessing, is as follows:

1. Train the YOLOv5 model on a development machine using PyTorch  
2. Export the trained model to ONNX format (`.onnx`) for optimized, framework-independent inference  
3. Transfer the ONNX model to the Raspberry Pi 5 and set up the runtime environment (Python, OpenCV, ONNX Runtime, NumPy)  
4. Start the inference server on the Raspberry Pi using ONNX Runtime with CPU execution and graph optimizations enabled  
5. On the client (laptop), capture webcam frames and compress them using JPEG (quality = 80) to reduce transmission size  
6. Send compressed frames over a TCP connection using a length-prefixed protocol to ensure reliable delivery  

7. **On the Raspberry Pi (Server-side processing):**  
   - Decode the incoming JPEG frame into an image  
   - Apply **letterbox preprocessing** to resize the image to 640×640 while preserving aspect ratio (padding added as needed)  
   - Normalize pixel values and convert the image into tensor format (NCHW) for model input  

8. **Model Inference:**  
   - Run the ONNX model using ONNX Runtime to generate raw predictions  
   - Output contains bounding boxes, objectness scores, and class probabilities  

9. **Postprocessing:**  
   - Apply **confidence thresholding** to filter out low-confidence detections  
   - Convert bounding boxes from center format (x, y, w, h) to corner format (x1, y1, x2, y2)  
   - Apply **Non-Maximum Suppression (NMS)** per class to remove overlapping detections and retain the most confident ones  
   - Map bounding boxes back to original image coordinates by reversing letterbox scaling and padding  

10. Package the final detections (class, confidence score, bounding box) into JSON format and send back to the client over TCP  

11. On the client side, receive detection results and render bounding boxes, class labels, and confidence scores on the original video frame  
12. Display real-time performance metrics such as FPS and inference latency for monitoring  

This pipeline ensures efficient preprocessing, optimized inference, and accurate postprocessing, enabling a stable end-to-end real-time detection system on edge hardware.
This deployment removes PyTorch dependency from the edge device and uses ONNX Runtime with graph optimizations and multi-threading for efficient CPU inference.

---

### On-Device Performance

| Metric | Observation |
|---|---|
| Inference latency | ~900–1000 ms per frame |
| Throughput | ~1–2 FPS |
| Resource utilization | CPU-based inference with multi-threading (4 threads) |
| Real-time behavior | Stable near real-time pipeline with continuous streaming |

The system maintains consistent performance during continuous operation, with stable latency and reliable frame processing. While the frame rate is limited due to CPU-only inference, the pipeline remains efficient and suitable for real-time monitoring use cases on edge devices.

## 8. System Prototype (Pictures / Figures)

The prototype demonstrates a complete distributed edge AI system with real-time interaction between a laptop and Raspberry Pi 5.

- **Hardware Setup:** Raspberry Pi 5 acts as the inference server, while the laptop handles frame capture and visualization  
- **Working Prototype:** Frames captured from a webcam are compressed and streamed to the Pi, where inference is performed and results are sent back  
- **Output Visualization:** Detected objects are displayed with bounding boxes, class labels, confidence scores, and real-time performance metrics (FPS, inference time)  

The system overlays runtime statistics such as FPS and inference latency directly on the output frames, enabling real-time monitoring of performance alongside detection results.

---

**Figure 11: Hardware setup showing the Raspberry Pi 5 (inference server) and laptop (client) used for real-time helmet detection.**

<img width="1600" height="879" alt="setup" src="https://github.com/user-attachments/assets/6b5ca8ba-eb79-462b-a776-daea0acdee25" />

---

**Figure 12: System pipeline illustrating the distributed client–server architecture, including frame capture, TCP transmission, ONNX-based inference, and result communication.**

<img width="1155" height="783" alt="pipeline" src="https://github.com/user-attachments/assets/c98be7a9-926b-489a-83b1-b07a1a5361c1" />

---

**Figure 13: Real-time detection output showing bounding boxes, class labels, confidence scores, and performance metrics (FPS and inference latency).**

<img width="556" height="368" alt="image" src="https://github.com/user-attachments/assets/cd248e38-90a0-4dfc-9f8d-db5f185256d5" />
<img width="758" height="373" alt="image" src="https://github.com/user-attachments/assets/533e22d6-88da-4d39-bc4e-870755d4e56d" />
<img width="634" height="305" alt="image" src="https://github.com/user-attachments/assets/8527e650-8bf1-4c7f-8a4a-296f4355aab3" />


## 9. Conclusions & Limitations

### Conclusion

A complete real-time helmet monitoring system was successfully implemented using a distributed edge AI architecture. The system demonstrates that low-cost hardware such as the Raspberry Pi 5 can perform real-time object detection using optimized inference pipelines without relying on cloud infrastructure.

The use of ONNX Runtime, TCP-based streaming, and efficient preprocessing enables a practical and scalable solution for edge deployment. The system is robust, modular, and capable of continuous real-time operation.

---

### Limitations

- Limited FPS due to CPU-only inference on the Raspberry Pi  
- Increased latency (~1 second per frame) restricts high-speed real-time applications  
- Detection performance can be affected in cases of heavy occlusion  
- Dependence on network stability for continuous TCP streaming  
- Small dataset limits robustness across diverse real-world scenarios 

## 10. Future Work

The system can be further improved in the following ways:

- Integrate hardware accelerators such as Coral TPU or Hailo for significant speed improvements  
- Apply model quantization (e.g., INT8) to reduce inference latency and improve throughput  
- Optimize the pipeline using parallel processing or asynchronous inference  
- Extend support for multiple camera streams for large-scale monitoring  
- Integrate number plate recognition (OCR) for complete traffic violation detection  
- Upgrade to newer architectures such as YOLOv8 or YOLO-NAS for improved accuracy and efficiency  

## 11. Challenges & Mitigation

### Challenges

- Running deep learning inference efficiently on resource-constrained hardware  
- Ensuring reliable transmission of image frames over TCP without data loss  
- Handling partial reads and incomplete data in socket communication  
- Managing large image payload sizes during real-time streaming  
- Debugging frame decoding issues and connection interruptions  

---

### Mitigation

- Used ONNX Runtime with graph optimizations and multi-threading for efficient CPU inference  
- Implemented a length-prefixed TCP protocol to guarantee complete message delivery  
- Applied JPEG compression (quality = 80) to reduce network bandwidth usage  
- Built robust socket handling with exact byte reads to prevent partial frame issues  
- Added error handling for decode failures and connection resets to maintain stability

## 12. Edge AI Optimization and Deployment Impact

This project incorporates key Edge AI principles by optimizing the trained model for deployment on a resource-constrained device (Raspberry Pi 5). Instead of running inference using PyTorch, the model was converted to ONNX format and executed using ONNX Runtime, enabling efficient CPU-based inference.

### Optimization Applied:
- Conversion from PyTorch (.pt) to ONNX (.onnx) for lightweight inference  
- Use of ONNX Runtime with graph optimizations enabled  
- Removal of PyTorch dependency on the edge device  
- JPEG compression (quality = 80) to reduce network transmission overhead  

---

### Before vs After Optimization Comparison

| Metric | Before Optimization (PyTorch - Training) | After Optimization (ONNX - Edge) |
|---|---|---|
| Runtime | PyTorch (GPU/desktop environment) | ONNX Runtime (CPU on Raspberry Pi 5) |
| Deployment feasibility | Not suitable for edge device | Fully deployable on Raspberry Pi |
| Precision | 0.9527 | ~same (no re-evaluation performed) |
| Recall | 0.8891 | ~same (expected unchanged) |
| mAP@0.5 | 0.9430 | ~same (expected unchanged) |
| Memory usage | High (PyTorch dependency) | Reduced (lightweight runtime) |
| Inference latency | Not measured on Pi | ~969 ms per frame |
| Throughput | Not practical on Pi | ~1.2 FPS |

---

### Observations:
- ONNX conversion preserves model weights and computation graph, so no significant accuracy degradation is expected.
- The primary improvement is in deployment feasibility and runtime efficiency rather than model accuracy.
- ONNX Runtime enables practical inference on Raspberry Pi by reducing memory usage and removing heavy dependencies.
- The system achieves stable near real-time performance (~1 FPS) on CPU-only hardware.
- JPEG compression further reduces network overhead, improving end-to-end latency.

This demonstrates how model optimization and runtime selection are critical for enabling real-time Edge AI applications on low-power devices.
