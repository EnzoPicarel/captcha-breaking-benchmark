<div align="center">
  <h3 align="center">Captcha Breaking Benchmark</h3>

  <p align="center">
   A comparative benchmark of <strong>Generative and Discriminative</strong> models implemented from scratch in <strong>NumPy</strong> to classify human vs. bot behavior.
    <br />
    <a href="#-getting-started"><strong>Quick Start »</strong></a>
  </p>
  
  ![CI Status](https://img.shields.io/badge/build-passing-brightgreen)
  ![License](https://img.shields.io/badge/license-MIT-blue)
</div>

## 🔍 About The Project
This project implements a complete supervised learning pipeline to classify users as Humans or Bots based on mouse interaction patterns (Captcha logs). The goal was to deconstruct the "black box" of machine learning libraries by implementing core algorithms entirely from scratch using only linear algebra.

Three distinct approaches are compared to solve this binary classification problem:
1.  **Probabilistic:** Gaussian Naive Bayes (Generative model estimating `P(X|y)`).
2.  **Statistical:** Logistic Regression (Linear Discriminative model optimizing Likelihood).
3.  **Neural Network:** Single-hidden-layer MLP (Non-linear approximation with Backpropagation).

*Built as a Data Science side project focusing on algorithmic implementation.*

### 🛠 Built With
* **Language:** Python 3.10+
* **Core Lib:** NumPy (Vectorized Matrix Operations)
* **Data Processing:** Pandas (ETL)

## 📐 Architecture

### Technical Highlights
* **Deep Learning from Scratch:** Full implementation of a neural network's forward and backward propagation steps (Chain Rule) without TensorFlow or PyTorch.
* **Vectorized Optimization:** Custom Gradient Descent and Likelihood Maximization algorithms utilizing NumPy broadcasting for performance.
* **Gaussian Density Estimation:** Manual calculation of log-probabilities and priors for the Naive Bayes classifier.
* **Reproducible Pipeline:** Seeded initialization (He Init for the neural network), Z-score normalization, and consistent train/test splitting logic.

### File Organization
```text
├── data/
│   └── captcha.csv         # Raw interaction logs (Is_Human bool, Mouse mvt, Time, etc.)
├── src/
│   ├── models/
│   │   ├── logistic.py     # Logistic Regression (Gradient Ascent)
│   │   ├── naive_bayes.py  # Gaussian Naive Bayes (Generative)
│   │   └── neural_net.py   # MLP (Manual Backprop & He Init)
│   └── utils.py            # Custom StandardScaler & Metrics
├── main.py                 # Benchmark orchestrator
└── requirements.txt        # Dependencies (numpy, pandas, tabulate)
```

## 🚀 Getting Started

### Prerequisites
* **Python 3.10+**
* **PIP**

### Installation & Build
1. **Clone and Setup Environment**
   ```bash
   # Clone the repository
   git clone https://github.com/enzopicarel/captcha-breaking-benchmark.git
   cd captcha-breaking-benchmark

   # Create and activate virtual environment
   python -m venv venv
   source venv/bin/activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## ⚡ Execution

Run the benchmark script to train all models sequentially and display the comparison table. Hyperparameters (learning rates, iterations, hidden size, epochs) can be changed in `main.py`.

```bash
python main.py
```

**Expected Output:**
```text
[1/3] loading data...
   samples: 70000 train, 30000 test

[2/3] benchmark start...
   -> training: Naive Bayes...
   -> training: Logistic Regression...
   -> training: Neural Network...

[3/3] results
| model               | accuracy   | precision   | recall   |
|:--------------------|:-----------|:------------|:---------|
| Naive Bayes         | 69.77%     | 72.25%      | 80.66%   |
| Logistic Regression | 70.32%     | 71.52%      | 84.05%   |
| Neural Network      | 70.28%     | 71.52%      | 83.96%   |
```

## 🧪 Tests
This project relies on the benchmark script itself for verification. A successful run is expected to show that:
1.  Training loss decreases for gradient-based models (Logistic Regression, MLP).
2.  Accuracy exceeds the random baseline (>50%).
3.  Implementations are numerically stable (e.g., sigmoid clipping).

## 👥 Authors
* **Enzo Picarel**

---
*Original Concept based on ENSEIRB-MATMECA Data Science Course.*
