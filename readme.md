# AI/ML Concepts & Implementation Notes

This document outlines key concepts in machine learning, computer vision, and AI system design, particularly in the context of real-world applications like facial recognition, document processing, and classification tasks.

## 🔧 Model Optimization Techniques

### 🎯 Fine-Tuning
- Fine-tuning adapts pre-trained models using custom data (e.g., company-specific terms, style guides).

### 🧠 Regularization
- **L1 / L2 Penalties**: Discourage large weights, reducing model complexity and overfitting.

### 📉 Dimension Reduction
- Removes irrelevant features, reducing sensitivity to noise in input data.

### ⏹️ Early Stopping
- Prevents over-learning of noise by halting training when performance plateaus.

## 🧍‍♂️ Face Recognition Pipeline

### 1. **Face Detection**
- Scans frames to locate face regions in images or videos.

### 2. **Face Identification (1:N)**
- Matches detected faces to a gallery of known individuals.
- Use-case: Classroom attendance, surveillance.

### 3. **Face Verification (1:1)**
- Confirms if a detected face matches a claimed identity.
- Use-case: Access control, secure login.

## 📚 Knowledge Mining & Language Understanding

### 🔍 Entity Extraction
- Extracts key pieces like names, dates, and locations from text.
- Operates at token or phrase level.

### 🏷️ Custom Named Entity Recognition (NER)
- Trained to recognize domain-specific entities that generic models miss.

### 🧾 Custom Text Classification
- Categorizes documents into user-defined labels.
- Ideal for use cases like ticket routing.
- Operates at document or paragraph level.

### 🧠 Prompt Structure
A prompt typically includes:
- **Instruction**: Task to perform
- **Context**: Relevant external information
- **Input Data**: The user’s query or text
- **Output Indicator**: Constraints (e.g., “Summarize in 3 sentences”)

## 🌍 Translation & OCR

### 🔁 Asynchronous Batch Translation
- Translates large files in bulk while preserving structure and formatting.

### ⚡ Synchronous Translation
- Fast, for single documents, but less suited for maintaining layout.

### 📖 Azure Read API
- Extracts text from cluttered or noisy images with high accuracy using deep learning.

### 🌗 Image Enhancement for OCR
- **Histogram Equalization**: Improves contrast in low-light conditions.
- **Histogram Normalization & HSV Analysis**: Adjust brightness based on ambient light.

## 📈 Evaluation Metrics

### ⚖️ Classification
- **F1 Score**: Best for imbalanced datasets.
- **Precision**: Prioritize when false positives are costly (e.g., financial fraud).
- **Recall**: Prioritize when false negatives are costly (e.g., spam detection).

### 📉 Regression
- **Mean Absolute Error (MAE)**: Reliable metric for continuous predictions.

### 🧪 Email Spam Detection Example:
- **True Positive**: Spam correctly flagged
- **True Negative**: Legit email received
- **False Positive**: Legit email flagged as spam
- **False Negative**: Spam delivered to inbox

## 🧠 Learning Types

### 🎓 Supervised Learning
- Requires labeled data (e.g., classification, regression).

### 🤖 Unsupervised Learning (Clustering)
- No labels required.
- Groups customers with similar patterns.
- Ideal for targeted marketing.

### 🏁 Reinforcement Learning
- Learns through trial-and-error with reward feedback.
- Not suitable for static data or clustering tasks.

## 🖼️ Computer Vision Concepts

### 🟪 Segmentation
- Identifies pixel regions (shapes), not instances.
- Used for outlining object shapes precisely.

### 🔲 Object Detection
- Locates and classifies multiple items via bounding boxes.
- Doesn’t give detailed shapes.

### 🧪 Image Processing Kernels
- **Emboss**: 3D shadow effect, not good for detail enhancement.
- **Edge Detection**: Highlights boundaries.
- **Blur**: Reduces noise but loses detail.
- **Sharpening**: Enhances edges and improves detail (best for license plate clarity).

## 🧠 Multi-Modal Models

- Combine image and text encoders (e.g., CNN + Transformer).
- Map embeddings into a shared vector space.
- Enable cross-modal tasks like image-captioning or visual Q&A.

## 🛠️ System Design Considerations

### 🔄 Controllability
- Model control decreases with increased complexity.

### 🗣️ Azure AI Services
- **Azure AI Translator**: Text translation only.
- **Azure AI Speech**: Real-time speech translation. Can be deployed with edge containers for security.
- **Azure AI Language**: Text processing (not speech).

### 📊 Interpretability
- Critical in healthcare or finance.
- Less essential in vision tasks where accuracy is prioritized.

### 🔐 Security
- Edge deployment allows sensitive data to stay local, improving security and compliance.

## ⚖️ Bias & Variance Trade-Off

> "Bias and variance cannot be optimized simultaneously" — True.  
Improving one often worsens the other. This balance is a core challenge in ML model design.

## 🧠 Hyperparameters

- Tunable settings that control the learning process (e.g., learning rate, batch size).

## 🧭 Guiding Principles

- **Preventing harm** is the top priority when deploying autonomous systems.

- **Traditional ML (e.g., Logistic Regression)** is efficient for small datasets with lower computation requirements, while deep learning is best for large-scale data and complex problems.

## 🔍 Azure Cognitive Search Integration

- Efficiently indexes large volumes of legal documents (e.g., court transcripts).
- Enables fast, relevant search and retrieval.
- Enhances accessibility and document management for legal workflows.

