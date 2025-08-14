# Generative AI Codedex 🚀

A comprehensive collection of Python implementations covering fundamental concepts in Generative AI and Natural Language Processing (NLP). This project serves as a practical learning resource for understanding key AI concepts through hands-on code examples.

## 📚 What You'll Learn

This project covers essential topics in Generative AI and NLP:

- **Tokenization** - Understanding how text is broken down into meaningful units
- **N-Grams** - Learning word sequence patterns for text prediction
- **Text Classification** - Building spam detection models using Naive Bayes
- **Machine Translation** - Converting text between different languages
- **Spell Checking & Sentiment Analysis** - Text correction and emotion detection
- **Advanced NLP with BERT** - State-of-the-art language model implementation

## 🛠️ Prerequisites

Before running this project, make sure you have:

- Python 3.7+ installed
- pip package manager
- Basic understanding of Python programming
- Interest in AI and machine learning

## 📦 Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/Generative_AI_Codedex.git
   cd Generative_AI_Codedex
   ```

2. **Install required dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

   Or install packages individually:

   ```bash
   pip install nltk numpy pandas scikit-learn translate textblob evaluate datasets huggingface-hub
   ```

## 🚀 Getting Started

### 1. Tokenization (`gen_ai_1.py`)

Learn how text is broken down into tokens for AI model training.

```bash
python gen_ai_1.py
```

**What you'll learn:**

- Understanding tokens and their importance in AI
- Using NLTK for text tokenization
- How language models process text data

### 2. N-Grams (`gen_ai_2.py`)

Explore word sequence patterns for text prediction and analysis.

```bash
python gen_ai_2.py
```

**What you'll learn:**

- Unigrams, bigrams, and trigrams
- Pattern recognition in text
- Probability-based text generation

### 3. Text Classification (`gen_ai_3.py`)

Build a spam detection system using machine learning.

```bash
python gen_ai_3.py
```

**What you'll learn:**

- Naive Bayes classification
- Text vectorization with CountVectorizer
- Model training and evaluation
- Real-world application: spam detection

**Note:** This script requires an `emails.csv` file. You can create your own dataset or modify the code to work with different text data.

### 4. Machine Translation (`gen_ai_4.py`)

Translate text between different languages automatically.

```bash
python gen_ai_4.py
```

**What you'll learn:**

- How machine translation works
- Using the `translate` library
- Language conversion techniques

### 5. Spell Checking & Sentiment Analysis (`gen_ai_5.py`)

Correct spelling mistakes and analyze text sentiment.

```bash
python gen_ai_5.py
```

**What you'll learn:**

- TextBlob library usage
- Automatic spell correction
- Sentiment analysis (positive/negative/neutral)

### 6. Advanced NLP with BERT (`gen_ai_6.ipynb`)

Jupyter notebook covering advanced sentiment analysis using BERT models.

```bash
jupyter notebook gen_ai_6.ipynb
```

**What you'll learn:**

- BERT model implementation
- Advanced sentiment analysis
- Data preprocessing for NLP tasks
- Model evaluation and metrics

## 📁 Project Structure

```
Generative_AI_Codedex/
├── gen_ai_1.py          # Tokenization fundamentals
├── gen_ai_2.py          # N-Grams and word patterns
├── gen_ai_3.py          # Text classification with Naive Bayes
├── gen_ai_4.py          # Machine translation
├── gen_ai_5.py          # Spell checking & sentiment analysis
├── gen_ai_6.ipynb       # Advanced NLP with BERT
├── README.md            # This file
├── requirements.txt     # Python dependencies
└── .gitignore          # Git ignore file
```

## 🔧 Dependencies

- **nltk** - Natural Language Toolkit for text processing
- **numpy** - Numerical computing
- **pandas** - Data manipulation and analysis
- **scikit-learn** - Machine learning algorithms
- **translate** - Machine translation library
- **textblob** - Text processing and analysis
- **evaluate** - Model evaluation metrics
- **datasets** - Hugging Face datasets
- **huggingface-hub** - Access to Hugging Face models

## 📖 Learning Path

For the best learning experience, follow this order:

1. **Start with** `gen_ai_1.py` - Understand the basics of tokenization
2. **Move to** `gen_ai_2.py` - Learn about word patterns and n-grams
3. **Progress to** `gen_ai_3.py` - Build your first ML model
4. **Explore** `gen_ai_4.py` - See machine translation in action
5. **Practice with** `gen_ai_5.py` - Text correction and sentiment analysis
6. **Master advanced concepts** with `gen_ai_6.ipynb` - BERT and modern NLP

## 🎯 Key Concepts Covered

- **Natural Language Processing (NLP)**
- **Machine Learning Fundamentals**
- **Text Preprocessing**
- **Model Training and Evaluation**
- **Real-world AI Applications**
- **State-of-the-art Language Models**

## 🙏 Acknowledgments

- **Codedex** for providing the learning platform
- **NLTK** team for the excellent natural language processing library
- **Scikit-learn** contributors for machine learning tools
- **Hugging Face** for state-of-the-art NLP models

