# 🧠 CogniFuse
## Dual-Encoder Fusion Framework for Mental Health Classification using BioClinicalBERT and RoBERTa

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

</p>

---

#  Overview

CogniFuse is a deep learning framework for multi-class mental health text classification that combines domain-specific clinical knowledge with general language understanding through a dual-encoder fusion architecture.

The framework utilizes:

- 🏥 BioClinicalBERT to capture clinical and biomedical language representations.
- 🌍 RoBERTa-base to learn contextual semantic information from general-domain text.
- 🔀 Feature-level fusion through concatenation.
- 🧠 A lightweight Multi-Layer Perceptron (MLP) classifier for final prediction.

The model classifies social media posts into four mental health categories:

- 😟 Anxiety
- 😔 Depression
- 😊 Normal
- ⚠️ Suicidal

The proposed architecture demonstrates improved classification performance compared to single-encoder baselines.

---

---

#  Features

- Dual-Encoder Transformer Architecture
- BioClinicalBERT Clinical Encoder
- RoBERTa General Language Encoder
- CLS Token Feature Extraction
- Feature Projection Layers
- Feature Concatenation Fusion
- MLP Classification Head
- Four-Class Mental Health Classification
- Extensive Evaluation Metrics
- Confusion Matrix
- ROC Curves
- Precision-Recall Curves
- t-SNE Embedding Visualization
- Ablation Study

---

#  Repository Structure

```
CogniFuse
│
├── figures/
│   ├── Architecture diagram.png
│   ├── Architecture diagram1.png
│   ├── confusion_matrix.png
│   ├── roc_curves.png
│   ├── precision_recall_curves.png
│   ├── tsne_embeddings.png
│   ├── eda_overview.png
│   └── ablation_results.png
│
├── notebooks/
│   └── code-paper-ai2ml-new.ipynb
│
├── results/
│   └── ablation_results.csv
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

# Dataset Overview

The dataset consists of mental health related social media posts categorized into four different psychological states.

Classes include:

- Anxiety
- Depression
- Normal
- Suicidal

The dataset exhibits class imbalance and varying sentence lengths, making it a challenging multi-class NLP classification problem.

<img width="2384" height="594" alt="eda_overview" src="https://github.com/user-attachments/assets/6d77346d-2246-43c7-8cd8-27bd66e36b1b" />

The figure above illustrates:

- Class distribution
- Word count distribution
- Average words per class

---

#  Model Architecture

The proposed CogniFuse framework consists of two parallel transformer encoders.

## Branch 1 — Clinical Encoder

- BioClinicalBERT
- Clinical domain knowledge
- CLS token extraction
- Linear Projection (768 → 256)

## Branch 2 — General Language Encoder

- RoBERTa-base
- General contextual representation
- CLS token extraction
- Linear Projection (768 → 256)

The projected feature vectors are concatenated into a 512-dimensional fused representation.

The fused embedding is then passed through an MLP classifier for final prediction.

<img width="1408" height="674" alt="Architecture diagram1" src="https://github.com/user-attachments/assets/65a76af0-d8f4-418c-adec-f9d565d600f5" />


---

---

#  Training Configuration

| Parameter | Value |
|------------|-------|
| Batch Size | 32 |
| Maximum Sequence Length | 128 |
| Projection Dimension | 256 |
| Fusion Dimension | 512 |
| Number of Classes | 4 |
| Optimizer | AdamW |
| Loss Function | CrossEntropy Loss |
| Framework | PyTorch |
| Transformers | HuggingFace |

---

#  Experimental Results

## Overall Performance

| Metric | Score |
|---------|-------|
| Accuracy | **86.2%** |
| Macro F1-score | **85.5%** |

---

#  Confusion Matrix

<img width="997" height="881" alt="confusion_matrix" src="https://github.com/user-attachments/assets/e9a40819-2fcb-4499-860c-99360c0683ae" />


The confusion matrix demonstrates strong classification performance across all four mental health categories, with the highest accuracy observed for the Normal class.

---

# ROC Curves

<img width="1034" height="881" alt="roc_curves" src="https://github.com/user-attachments/assets/6cb85876-98bf-4fa1-a87e-59b7de184985" />


Area Under Curve (AUC):

| Class | AUC |
|--------|-----|
| Anxiety | 0.991 |
| Depression | 0.949 |
| Normal | 0.994 |
| Suicidal | 0.959 |

---

# Precision–Recall Curves

<img width="1034" height="881" alt="precision_recall_curves" src="https://github.com/user-attachments/assets/635e4c7a-229b-4506-bfff-03714a807094" />


Average Precision (AP):

| Class | AP |
|--------|----|
| Anxiety | 0.948 |
| Depression | 0.890 |
| Normal | 0.993 |
| Suicidal | 0.874 |

---

#  Embedding Visualization (t-SNE)

<img width="1184" height="1031" alt="tsne_embeddings" src="https://github.com/user-attachments/assets/fd2b872a-8890-4b87-a53a-5685206bd067" />

The t-SNE projection demonstrates that the fused feature representations form well-separated clusters for each mental health category, indicating that the dual-encoder architecture learns highly discriminative embeddings.

---

#  Ablation Study

To evaluate the effectiveness of feature fusion, the proposed model was compared against single-encoder baselines.

<img width="1784" height="741" alt="ablation_results" src="https://github.com/user-attachments/assets/c0a474b9-9d97-4248-9c6f-71ceeaf1748a" />


| Model | Accuracy | Macro F1 |
|--------|----------|----------|
| ClinicalBERT Only | 81.8% | 80.8% |
| MentalBERT Only | 74.2% | 72.5% |
| **CogniFuse (Dual Encoder)** | **86.2%** | **85.5%** |

The dual-encoder fusion architecture consistently outperformed both individual encoders, demonstrating the complementary strengths of clinical and general-domain language representations.

---

#  Installation

Clone the repository:

```bash
git clone https://github.com/Codennmu/CogniFuse.git

cd CogniFuse
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

#  Usage

Open the Jupyter notebook:

```bash
jupyter notebook
```

Run:

```
code-paper-ai2ml-new.ipynb
```

The notebook includes:

- Data preprocessing
- Model construction
- Training
- Evaluation
- Visualization
- Ablation study

---

#  Future Work

Potential future improvements include:

- Multi-modal mental health detection
- Explainable AI (XAI)
- Knowledge Graph Integration
- Cross-lingual Mental Health Detection
- Continual Learning
- Clinical Deployment
- Federated Learning

---

#  Citation

If you use this repository in your research, please cite:

```bibtex
@article{CogniFuse2026,
  title={CogniFuse: Dual-Encoder Fusion Framework for Mental Health Classification},
  author={Saad Mazher},
  journal={Under Review},
  year={2026}
}
```

---

# 📄 License

This project is released under the MIT License.

---


---

## ⭐ If you find this repository useful, please consider giving it a Star.
