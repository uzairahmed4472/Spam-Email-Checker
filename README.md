# Spam Email Detection System

A complete machine learning pipeline for detecting spam emails using LSTM neural networks. This system includes data preprocessing, model training, and a REST API for real-time predictions.

## 📁 Project Structure

```
project/
├── app.py              # Flask REST API for spam detection
├── preprocess.py       # Data preprocessing and cleaning
├── train.py           # LSTM model training script
├── utils_text.py      # Text cleaning utilities
├── data/
│   ├── Emails.csv     # Raw email dataset
│   └── processed.csv  # Cleaned and balanced dataset
└── artifacts/         # Saved model files (created after training)
    ├── spam_model.keras
    ├── tokenizer.pkl
    └── config.json
```

## 🚀 How It Works

### 1. Data Preprocessing (`preprocess.py`)

- **Loads raw email data** from `Emails.csv`
- **Cleans text** by removing:
  - Email headers and metadata
  - URLs and email addresses
  - Numbers and punctuation
  - Common stopwords
- **Balances the dataset** by downsampling the majority class (ham)
- **Saves processed data** to `processed.csv`

### 2. Model Training (`train.py`)

- **Tokenizes text** into sequences of numbers
- **Builds LSTM neural network** with:
  - Embedding layer (converts words to vectors)
  - LSTM layer (learns sequential patterns)
  - Dense layers (classification)
- **Trains the model** with early stopping and learning rate reduction
- **Evaluates performance** using accuracy, precision, recall, and ROC AUC
- **Saves model artifacts** for deployment

### 3. API Service (`app.py`)

- **Loads trained model** and tokenizer at startup
- **Provides REST endpoints**:
  - `GET /` - Health check
  - `POST /predict` - Spam detection
- **Processes requests** by cleaning text and making predictions
- **Returns JSON responses** with classification and confidence scores

## 🛠️ Installation & Setup

### Prerequisites

```bash
pip install pandas numpy scikit-learn tensorflow nltk flask waitress
```

### Step 1: Prepare Data

```bash
python preprocess.py
```

This will:

- Clean the raw email data
- Balance spam/ham classes
- Save processed data to `data/processed.csv`

### Step 2: Train Model

```bash
python train.py
```

This will:

- Train the LSTM model
- Evaluate performance
- Save model artifacts to `artifacts/` folder

### Step 3: Start API Server

```bash
python app.py
```

The server will start on `http://localhost:8000`

## 📡 API Usage

### Health Check

```bash
curl http://localhost:8000/
```

Response:

```json
{ "status": "ok", "service": "spam-detector" }
```

### Spam Detection

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Congratulations! You have won $1000! Click here to claim your prize!"}'
```

Response:

```json
{
  "input": "Congratulations! You have won $1000! Click here to claim your prize!",
  "cleaned": "congratulations won click claim prize",
  "label": "Spam",
  "confidence": 0.9876
}
```

## 🧠 Model Architecture

The LSTM model consists of:

1. **Embedding Layer**: Converts word indices to dense vectors (32 dimensions)
2. **LSTM Layer**: 16 units for sequential pattern learning
3. **Dense Layer**: 32 neurons with ReLU activation
4. **Output Layer**: Single neuron with sigmoid activation (0-1 probability)

### Hyperparameters

- **Max sequence length**: 100 words
- **Vocabulary size**: 20,000 most frequent words
- **Batch size**: 32
- **Epochs**: 12 (with early stopping)
- **Optimizer**: Adam
- **Loss function**: Binary crossentropy

## 📊 Performance Metrics

The model is evaluated using:

- **Accuracy**: Overall correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **ROC AUC**: Area under the ROC curve

## 🔧 Configuration

Key parameters can be adjusted in `train.py`:

- `MAX_LEN`: Maximum words per email
- `VOCAB_SIZE`: Vocabulary size
- `epochs`: Training epochs
- `batch_size`: Training batch size

## 📝 Code Comments

All code files include detailed line-by-line comments explaining:

- **What each line does**
- **Why it's necessary**
- **How it fits into the overall pipeline**
- **Technical details and parameters**

This makes the code educational and easy to understand for learning purposes.

## 🚨 Error Handling

The system includes robust error handling for:

- Missing data files
- Invalid input formats
- Model loading failures
- Network connectivity issues

## 🔄 Workflow Summary

1. **Raw Data** → `preprocess.py` → **Clean Data**
2. **Clean Data** → `train.py` → **Trained Model**
3. **Trained Model** → `app.py` → **Live API**
4. **Email Text** → **API** → **Spam/Ham Classification**

This complete pipeline demonstrates a real-world machine learning application from data preprocessing to deployment.
