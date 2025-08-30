# User Guide
## 1. Evaluate Model Performance
Use `barbie_eval.ipynb` to assess the model using a labeled dataset.
- **Input**: CSV file with two columns: `['cleaned_text', 'classification']`
- **Output**: Evaluation metrics (e.g., accuracy, precision, recall)
## 2. Label Unlabeled Reviews
Use `barbie_label.ipynb` to automatically label raw review data.
- **Input**: CSV file with one column: `['cleaned_text']`
- **Output**: CSV file with added `classification` column (binary labels)
## 3. Train the Model
To train the model from scratch:

- **Step 1**: Use `clean.py` to convert your raw JSON dataset into a cleaned CSV
- **Step 2**: Run `initial_label.ipynb` to generate initial labels for training
- **Step 3**: Load the labeled dataset into `main.ipynb` to begin training

# Project Overview
## Introduction
### 1. Problem Statement
Online reviews are vital for consumer decisions, but platforms are overwhelmed by irrelevant, misleading, and low-quality content. Existing filters rely on basic keyword detection, failing to capture context or enforce quality—resulting in spam, off-topic posts, and fake reviews.
### 2. Research Question
Can Machine Learning and Natural Language Processing automatically assess review quality and relevance to build a more trustworthy review ecosystem?
### 3. Project Impact
- **Users**: Improved trust in reviews → smarter choices on where to go and spend
- **Businesses**: Fairer representation by removing misleading or malicious reviews
- **Platforms**: Scalable, automated moderation → reduced manual effort and higher credibility
### 4. Our Apporoach
We apply advanced NLP and metadata analysis to identify genuine, context-rich reviews—moving beyond keyword matching to restore trust in location-based platforms.

## Project Timeline
**Preparation**
- Set up Python 3.13 environment with required libraries such as numpy, pandas, jupyter etc.
- Reviewed dataset structure and clarified labeling categories.
- Researched suitable ML/NLP models for the problem.
- Assigned roles across the team (data, modeling, reporting).

**Data Processing**
- Collected and cleaned dataset (removed duplicates, handled missing entries).
- Generated pseudo-labels to label reviews into predefined categories.
- Split data into training and validation sets.

**Model Definition & Training**
- Implemented baseline model (Logistic Regression).
- Built BERT-based classification pipeline using Hugging Face Transformers.
- Trained BERT on the labeled dataset and experimented with few-shot prompting.

**Testing & Refinement**
- Evaluated results with precision, recall, F1-score.
- Conducted error analysis to identify misclassified categories.
- Refined prompts, adjusted thresholds, and tuned hyperparameters.

**Finalization & Deliverables**
- Selected the best-performing model.
- Created visualizations for evaluation metrics.
- Compiled results into a final report and presentation deck.
- Prepared project repository for reproducibility (requirements.txt + README).





## Model Architecture
### 1. Model Choice
We chose **BERT** (bert-based-uncased) as the backbone for our review relevance classification task.

**Why BERT?**
- **Pretrained Knowledge**: BERT has already been trained on a very large text corpus (Wikipedia + Books). This gives it a “head start” in understanding English before we fine-tune it on our task.
- **Contextual Understanding**: Unlike older models (like bag-of-words or TF-IDF), BERT understands context. For example:
    - “The food is **not good**” ≠ “The food is good”.
    - BERT knows the word _not changes_ the meaning.
- **WordPiece Tokenization**: BERT can split unknown words into subwords.
    - Example: “pizzapromo” → “pizza” + “##promo”.
    - This means it can handle spelling variations, slang, and new words.
- **Classification Head**: On top of BERT, we add one simple linear layer that takes the `[CLS]` token (the special “summary” token) and predicts whether the review is **0(irrelevant)** or **1 (relevant)**.

**Advantages:** 
- Strong accuracy
- Good at handling context
- Working well with limited labeled data

**Disadvantages:** 
- Large model (~110M parameters)
- Slower training
- Need GPU for efficiency.

### 2. Data Preprocessing
**1. Text Cleaning**
- Lowercased all text (since we use `bert-base-uncased`).
- Removed URLs, extra spaces, and non-informative symbols.
- _Why_: light cleaning preserves semantics, avoids harming BERT’s tokenizer.

**2. Tokenization & Encoding**
- Used BERT WordPiece tokenizer.
- Added special tokens `[CLS]` + `[SEP]`.
- Applied padding/truncation to `max_length=128`.
- Generated **attention masks** (1 = token, 0 = padding).
- _Why_: ensures consistent tensor shape for batching.

**3. Train/Validation Split**
- 80/20 split with fixed random seed for reproducibility.
- (Optional: stratify by label if data is imbalanced).
- _Why_: prevents overfitting checks, makes results consistent.

**4. Conversion to Tensors**
- Converted `input_ids, attention_mask`, and `labels` into PyTorch tensors.
- _Why_: required format for BERT and DataLoader batching.

**Advantages:**
- Reproducible
- Efficient
- Compatible with BERT’s expected input

**Disadvantages:** 
- Long reviews truncated at 128 tokens,
- Padding increases compute slightly.


### 3. Hyperparameter Tuning
Hyperparameters are “settings” we choose before training (they are not learned by the model). They control how the model learns.

**Our Tuning Process**
- **Baseline Setup (Default Fine-tuning)**
    - **Learning Rate**: `2e - 5 `(very small to avoid overwriting BERT’s pretrained knowledge).
    - **Batch Size**: `32` (process 32 reviews at once).
    - **Epochs**: `3` (3 full passes through the dataset).
    - **Optimizer**: `AdamW` (variant of Adam that prevents overfitting with weight decay).
    - **Scheduler**: Linear decay with warmup (learning rate starts higher, then gradually decreases).
- **Exploration (Hyperparameter Search)**
    - Tested different **learning rates**: `{1e-5, 2e-5, 3e-5, 5e-5}`.
    - Tried different **batch sizes**: `{16, 32}`.
    - Compared **max sequence lengths**: `{128, 256}`tokens (shorter vs. longer reviews).
    - **Method:**
        - Started with **Random Search** (fast way to test different combinations without trying every possibility).
        - Then used **Bayesian Optimization** (Optuna) to fine-tune within the best ranges.

**Advantages:**
- Efficient search
- Avoid wasting time on bad combinations 

**Disadvantages:** 
- Still computationally expensive  tuning takes time

### 4. Training Loop
The training process is a cycle of feeding batches of reviews into the model, adjusting weights, and then checking performance.
#### Steps:
**1. Training Mode**
- Model set to `model.train()` → enables dropout (helps avoid overfitting).
- Loop through batches from `train_dataloader`.

**2. Forward Pass**
- Send inputs (`input_ids, attention_mask, labels`) into BERT.
- Get **logits** (raw predictions) and **loss** (how wrong the predictions are).

**3. Backward Pass**
- Call `.backward()` → PyTorch computes gradients for each parameter.
- Clip gradients (`clip_grad_norm_`) at 1.0 to avoid exploding updates.
- Update weights with `optimizer.step()`.
- Adjust learning rate with `scheduler.step()`.

**4. Validation Mode**
- Switch to `model.eval()` (no dropout, more stable predictions).
- Compute predictions on `validation_dataloader`.
- Compare predictions with true labels → calculate **validation accuracy**.

**Advantages:** 
- Stable training
- Prevent overfitting
- Track performance each epoch

**Disadvantages:** 
- Accuracy may not be enough if data is imbalanced , so we should also track **precision, recall, and F1**.

### 5.Pipeline Diagram
Here’s a simplified view of how raw text becomes model input:
```
Review (text)
      ↓
Tokenizer (WordPiece) → splits into tokens
       ↓
Token IDs (integers from BERT’s vocab)
       ↓
Tensor (fixed length, padded + attention mask)
       ↓
BERT Model (bert-base-uncased + classification head)
       ↓
Prediction (Relevant = 1 / Irrelevant = 0)
```

# Setup Instructions
## Data Labeling Setup
### 1. Overview
We use a **zero-shot classification** approach via Hugging Face’s `facebook/bart-large-mnli` model to automate binary labeling of location reviews—eliminating the need for manual annotation.

### 2. Prerequisites
`pip install transformers torch pandas numpy`

### 3. Labeling Schema
- **1 = Relevant**: Authentic customer experience, detailed visit/service description
- **0 = Irrelevant**: Spam, ads, off-topic, fabricated, low-detail, gibberish

### 4. Setup Instructions
**Step 1**: Prepare CSV with `review_text` (plus optional fields: rating, location, timestamp) 

**Step 2**: Install dependencies 

**Step 3**: Run labeling script using zero-shot classifier 

**Step 4**: Spot-check ~100–200 samples for accuracy; refine candidate labels if needed 

**Step 5**: Final output saved as `train_final.csv`

### 5. Labeling Criteria
**Relevant (1):**
- Genuine visit descriptions
- Constructive feedback
- Detailed service/atmosphere notes

**Irrelevant (0):**
- Ads, promotions, external links
- Off-topic or one-word responses
- Fake/gibberish content
- Mixed relevance (e.g., review + promo)

### 6. Troubleshooting
- **Memory issues**: Process in batches
- **Model errors**: Check internet connection
- **Inconsistent labels**: Refine candidate phrases or add preprocessing

**Advantages:** 
- Fast, scalable labeling without manual effort
- Context-aware classification using NLP
- Easy integration with downstream ML pipelines

**Disadvantages:**
- May misclassify edge cases or ambiguous reviews
- Requires manual validation for quality assurance
- Limited nuance in binary schema (no gradation of relevance)

# How to Reproduce Results
## Environment Setup
- **Hardware Used:**
    - CPU : AMD Ryzen 7 5800H
    - GPU: NVIDIA GeForce RTX 3050 (Laptop)
- **Python Version**: Python 3.13
- **Requirements.txt Includes**:
    - `numpy, pandas, scikit-learn` - data preparation & data processing
    - `torch` - deep learning framework
    - `jupyter` - interactive experimentation
- **Tools Used:**
    - **Jupyter Notebook**: for model training, data labeling, and visualization.
        - **Pros:**
            1. Interactive cell-by-cell execution → faster prototyping & debugging.
            2. Inline visualization of metrics, plots, confusion matrices.
            3. Great for hackathons: quick experimentation and easy presentation.
        - **Cons:**
            1. Not ideal for production-scale code.
            2. Execution order can cause confusion if cells are run inconsistently.
    - **PyCharm**: for offline data preparation scripts (cleaning, preprocessing).
        - **Pros:**
            1. Strong debugging, code refactoring, and linting support.
            2. Encourages modular and reusable scripts for data pipelines.
            3. Built-in Git integration for teamwork.
        - **Cons:**
            1. Heavier than lightweight editors.
            2. Less interactive than Jupyter for rapid experiments.

## Analyzing Output
To evaluate our BERT-based classification model, we used multiple performance metrics to capture both overall accuracy and class-level performance:

- **Accuracy (0.978)**: While relatively high, we believe accuracy can be misleading in imbalanced datasets, since predicting the majority class alone may yield inflated results. Therefore, we focus more on precision, recall, and F1.
- **Precision (0.992)**: Indicates the model is highly effective in avoiding false positives.
- **Recall (0.985)**: Shows that the model successfully captures the majority of relevant cases. 
- **F1 Score (0.989)**: Balances precision and recall, reflecting strong overall reliability.

**Labelling Output**
Beyond evaluation, we tested the model in a practical setting by feeding it unlabelled cleaned reviews. The model was able to automatically assign relevant/irrelevant labels with high consistency. This demonstrates its potential utility in real-world applications where large volumes of user reviews need to be processed and classified efficiently.
