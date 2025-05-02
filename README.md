# Sentiment-Analysis-on-Movie-Reviews

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![NLP](https://img.shields.io/badge/Natural-Language_Processing-green)
![Transformers](https://img.shields.io/badge/Transformers-BERT%2FSBERT-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

A comprehensive comparison of NLP techniques for sentiment analysis on movie reviews, ranging from traditional methods to state-of-the-art transformer models.

## Table of Contents
- [Overview](#-overview)
- [Features](#-features)
- [Techniques Implemented](#-techniques-implemented)
- [Results](#-results)
- [License](#-license)

## Overview

This project explores and compares multiple NLP approaches for binary sentiment classification (positive/negative) on movie reviews, providing insights into:
- Traditional NLP techniques vs modern deep learning methods
- The evolution of text representation from BoW to contextual embeddings
- Performance trade-offs between different approaches

## Features

- **Multiple NLP Techniques** compared in one framework
- **Ready-to-use** implementations of various text representation methods
- **Pre-trained models** included for BERT and SBERT
- **Modular code** for easy experimentation
- **Detailed evaluation** metrics for each approach

## Techniques Implemented

### **1. Traditional NLP Approaches**
| Method        | Implementation Details | Key Parameters |
|--------------|-----------------------|----------------|
| **Bag-of-Words (BoW)** | CountVectorizer from scikit-learn | `max_features=5000`, `ngram_range=(1,2)` |
| **TF-IDF** | TfidfVectorizer from scikit-learn | `max_df=0.85`, `min_df=5`, `sublinear_tf=True` |

*Classifier Used*: Logistic Regression with `max_iter=1000`, `C=1.0`

### **2. Word Embeddings**
| Method       | Implementation | Pretrained Model | Dimensionality |
|--------------|----------------|------------------|----------------|
| **Word2Vec** | Gensim library | Custom-trained on corpus | 300-dim vectors |
| **BERT** | `transformers` library | `bert-base-uncased` | 768-dim vectors |
| **SBERT** | `sentence-transformers` | `all-MiniLM-L6-v2` | 384-dim vectors |

*Feature Extraction*: Mean pooling of token embeddings for document-level representation

### **3. Advanced Models**
| Model | Architecture | Training Details |
|-------|-------------|------------------|
| **Fine-tuned BERT** | `BertForSequenceClassification` | 3 epochs, `batch_size=16`, `lr=2e-5` |


---

## Results

### **Performance Comparison** (IMDb Test Set)
| Model | Accuracy | 
|-------|----------|
| BoW + LogisticReg | 86.8% | 
| TF-IDF + LogisticReg | 88.6% |
| Word2Vec + LR | 87.4% | 
| BERT Embeddings + LR | 88.9% | 
| SBERT Embeddings + LR | 90.8% | 
| **Fine-tuned BERT** | **94.8%** |

*\*On Google Colab (T4 GPU for deep learning models)*

### **Key Observations**
1. **Traditional vs Deep Learning**:
   - TF-IDF outperforms Word2Vec (88.6% vs 87.4%)
   - Contextual embeddings (BERT/SBERT) show ~3% improvement over TF-IDF

2. **Transformer Advantage**:
   - Fine-tuned BERT achieves **94.8% accuracy** (best overall)

3. **Error Analysis**:
   - Common misclassifications occur with:
     - Sarcastic reviews (e.g., "Oh great, another superhero movie...")
     - Mixed sentiments (e.g., "The acting was superb but the plot was weak")

## License  
This project is licensed under the [MIT License](LICENSE) - see the [LICENSE](LICENSE) file for details.
