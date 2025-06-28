
# Implicit Feedback Recommender System

A production-ready recommendation system that learns from user interactions like views, clicks, and purchases.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Features

- **Multiple Algorithms**: ALS, BPR, and Logistic Matrix Factorization
- **Implicit Feedback**: Works with views, clicks, purchases
- **Cold Start Support**: Handles new users and items
- **Explainable Recommendations**: Provides reasoning for suggestions
- **Evaluation Framework**: Precision@K, Recall@K, MAP@K metrics
- **Web Interface**: Simple Flask-based API

## Installation

### Prerequisites

- Python 3.8+
- pip 20+

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/implicit-recommender.git
   cd implicit-recommender

2. Install dependencies:

    ```bash
    pip install -r requirements.txt

3. Generate sample data:

    ```bash
    python -m scripts.train_model

## Usage

### 1. Training the Model

    python -m scripts.train_model

### 2. Starting the API Serving
    
    python -m app.views

### 3. Evaluating the Model

    python -m scripts.evaluate_model

