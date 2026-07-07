# 📚 arXiv Article Classifier: Scientific Paper Topic Classification

[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://arxiv-classifier-7mdehfxzzdxg9usumpauq7.streamlit.app/)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-ffd21e)](https://huggingface.co/sofia-ol/arxiv-classifier)

An interactive web application for classifying scientific papers from the arXiv repository by their topic area based on title and abstract. The service accepts article text input and returns the most likely categories sorted by probability until reaching a cumulative threshold of 95%.

## 📋 Description

This project implements a text classification system for scientific articles using DistilBERT, a lightweight transformer model. The classifier is fine-tuned on a custom dataset of arXiv papers across multiple disciplines including Computer Science, Physics, Mathematics, and Biology.

**Key Features:**
- 📝 Classifies papers based on title and abstract
- 📊 Returns top categories with confidence scores
- 🎯 Configurable confidence threshold (default 95%)
- ⚡ Fast inference with DistilBERT
- 🌐 Deployed as a web application with Streamlit

## ✨ Features

- **Multiple Categories:** Supports 7 scientific domains (AI, NLP, CV, ML, Physics, Math, Biology)
- **Interactive Interface:** Clean, responsive UI with progress bars
- **Configurable Threshold:** Adjustable confidence threshold (50-99%)
- **Cached Model:** Fast loading using `@st.cache_resource`
- **Error Handling:** Validates input and handles model errors gracefully
- **Visual Feedback:** Clear visualization with category icons and confidence bars

## 🔧 Technologies

- **Machine Learning Framework:** PyTorch
- **Base Model:** DistilBERT-base-uncased (HuggingFace Transformers)
- **Web Framework:** Streamlit
- **Deployment:** Streamlit Community Cloud
- **Model Hosting:** Hugging Face Hub
- **Data Processing:** NumPy

## 📊 Dataset

### Data Collection

Data was collected using the official arXiv API from the following categories:
- `cs.AI` (Artificial Intelligence)
- `cs.CL` (Computation & Language)
- `cs.CV` (Computer Vision)
- `cs.LG` (Machine Learning)
- `physics` (Physics)
- `math` (Mathematics)
- `q-bio` (Quantitative Biology)

### Data Preparation

- Total samples: ~15,000 articles
- Each sample: `"Title [SEP] Abstract"`
- Split: 80% training, 10% validation, 10% test

## 🧪 Model and Experiments

### Base Architecture

**DistilBERT-base-uncased** - A lightweight version of BERT that provides an excellent balance between inference speed and classification quality.

### Hyperparameter Experiments

| Experiment | Learning Rate | Batch Size | Epochs | Weight Decay | Validation Accuracy |
|------------|---------------|------------|--------|--------------|---------------------|
| **Baseline** | **2e-5** | **16** | **3** | **0.01** | **74.9%** |
| More Epochs | 2e-5 | 16 | 5 (early stop) | 0.01 | 73.5% |
| High Regularization | 2e-5 | 16 | 3 | 0.1 | 72.8% |
| Large Batch | 2e-5 | 32 | 3 | 0.01 | 71.4% |
| Smaller LR | 1e-5 | 16 | 3 | 0.01 | 71.1% |

### Best Configuration

- **Learning Rate:** 2e-5
- **Batch Size:** 16
- **Epochs:** 3
- **Weight Decay:** 0.01
- **Validation Accuracy:** 74.9%

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Clone the Repository

```bash
git clone https://github.com/your_username/arxiv-classifier.git
cd arxiv-classifier
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Locally

```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`

## 📱 Usage

1. **Enter Article Title:** Required field - enter the paper title
2. **Enter Abstract:** Optional field - paste the abstract
3. **Adjust Threshold:** Use the slider to set confidence threshold (default 95%)
4. **Click Classify:** Press the "CLASSIFY" button
5. **View Results:** See top categories with confidence bars

### Example Input

**Title:** "Attention is All You Need"
**Abstract:** "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks..."

### Example Output

```
📊 Classification Results

🤖 Artificial Intelligence
███████████████████████████████████████ 73.2% confidence

📊 Machine Learning
███████████████████████████████████████████ 21.7% confidence

💬 Computation & Language
████████████████ 5.1% confidence
```

## 🏗️ Architecture

### System Workflow

```
User Input (Title + Abstract)
    ↓
Text Preprocessing ("Title [SEP] Abstract")
    ↓
Tokenization (max_length=384)
    ↓
DistilBERT Model Inference
    ↓
Softmax Probabilities
    ↓
Top Categories (cumulative ≥ threshold)
    ↓
Display Results with Confidence Bars
```

### Key Components

**1. Model Loading (`load_model`)**
- Caches model using `@st.cache_resource`
- Loads from Hugging Face Hub
- Moves to GPU if available

**2. Prediction (`predict`)**
- Tokenizes input text
- Runs model inference
- Sorts categories by probability
- Filters to top probabilities until 95% cumulative

**3. UI Components**
- Title and abstract input fields
- Confidence threshold slider
- Category display with progress bars
- Result interpretation

## 📤 Output Format

The application displays results with:

1. **Category Name:** Human-readable name with icon
2. **Confidence Bar:** Visual progress bar with percentage
3. **Category Code:** arXiv category code (e.g., `cs.AI`)
4. **Summary Statement:** 
   - Single category: "Article clearly belongs to [Category]"
   - Multiple categories: "Article covers N areas (total probability > X%)"

## 🔄 Deployment

### Streamlit Cloud

The application is deployed on Streamlit Community Cloud and is available at:

**[https://arxiv-classifier-7mdehfxzzdxg9usumpauq7.streamlit.app/](https://arxiv-classifier-7mdehfxzzdxg9usumpauq7.streamlit.app/)**

### Deployment Steps

1. Push code to GitHub repository
2. Connect repository to Streamlit Cloud
3. Configure `requirements.txt` with dependencies
4. Deploy with one-click

### Model Hosting

The fine-tuned DistilBERT model is hosted on Hugging Face Hub:
- **Repository:** [sofia-ol/arxiv-classifier](https://huggingface.co/sofia-ol/arxiv-classifier)
- **Framework:** PyTorch
- **Format:** Model weights + tokenizer config

## 📊 Performance

| Metric | Value |
|--------|-------|
| **Validation Accuracy** | 74.9% |
| **Number of Categories** | 7 |
| **Training Samples** | ~15,000 |
| **Model Size** | ~267 MB |
| **Inference Time** | < 1s (cached) |

## 🧠 Model Details

### Base Model

- **Name:** DistilBERT-base-uncased
- **Parameters:** 66 million
- **Type:** Distilled version of BERT
- **Advantages:** 40% faster than BERT, 97% of BERT's performance

## 🐛 Error Handling

The application handles various error scenarios:

1. **Empty Input:** Validates that title is not empty
2. **Model Loading:** Catches and displays loading errors
3. **Prediction Errors:** Gracefully handles model inference issues
4. **Missing Model:** Provides clear error message when model not found

## 📈 Future Development

- Support for more arXiv categories
- Batch classification for multiple papers
- Keyword extraction from abstracts
- Confidence calibration
- Integration with arXiv API for direct paper retrieval
- Support for other languages
- Ensemble methods for improved accuracy
- Real-time paper recommendation based on topic

## 🎨 Customization

### Adding New Categories

To add a new category:

1. Update `CATEGORY_NAMES` dictionary
2. Extend the dataset with new category samples
3. Retrain the model with updated labels
4. Upload new model to Hugging Face Hub

### Adjusting Threshold

The confidence threshold can be adjusted via the slider in the UI:
- **Lower threshold:** Shows more categories (lower confidence)
- **Higher threshold:** Shows fewer categories (higher confidence)
- 
---

**Live Demo:** [arXiv Article Classifier](https://arxiv-classifier-7mdehfxzzdxg9usumpauq7.streamlit.app/)
