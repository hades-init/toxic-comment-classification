# Toxic Comment Classification

\
\
This project implements a toxic comment classification system using machine learning techniques, trained on the [Jigsaw Toxic Comment Classification Challenge dataset](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge). The system predicts whether a comment is toxic across six categories: `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, and `identity_hate`. A web interface allows users to input comments and view predicted scores for each label.

### Approaches Used

Three distinct approaches were explored to solve this multi-label classification problem:

#### TF-IDF with Traditional ML:
*Description*: Text is preprocessed and converted to TF-IDF vectors, capturing term frequency and inverse document frequency. Models like Random Forest, Logistic Regression, and Naive Bayes were trained on these features.
*Pros*: Simple, interpretable, and effective with sparse data.
*Cons*: Lacks semantic understanding and context.

#### Word Embeddings (GloVe):
*Description*: Pre-trained GloVe embeddings (100-D) were used to represent words, aggregated (averaged or concatenated) into sentence vectors. Random Forest and other ML models were applied.\
*Pros*: Captures semantic relationships between words.\
*Cons*: Aggregation loses word order and specific toxic signals, underperforming compared to TF-IDF.

#### Transformer-Based Models (DistilBERT):
*Description*: Fine-tuned a pre-trained RoBERTa-base model ([twitter-roberta-base checkpoint](https://huggingface.co/cardiffnlp/twitter-roberta-base)) for multi-label classification. The transformer processes raw text, leveraging contextual embeddings for superior understanding. \
*Pros*: State-of-the-art performance, captures context and nuance. \
*Cons*: Computationally intensive, requires GPU for efficient training.

### Performance Comparison

The approaches were evaluated based on F1-scores for the toxic class (and overall multi-label metrics where applicable):

**Transformers (RoBERTa-base)**: Best performance, with F1-scores ~0.75–0.85 for toxic and strong results across all labels after fine-tuning. Excels due to contextual understanding. \
**TF-IDF**: Moderate performance, F1-scores ~0.5–0.65. Effective for frequent labels like toxic but struggles with rare ones (e.g., threat). \
**Word Embeddings (GloVe)**: Lowest performance, F1-scores <0.5. Semantic richness didn’t translate to better results with Random Forest, likely due to information loss in averaging and lot of out-of-vocabulary words (e.g. slangs and slurs).

### Conclusion: 
Transformers outperformed TF-IDF, which in turn outperformed GloVe embeddings, making Transformers (RoBERTa-base) the chosen approach for deployment.


### Setup Instructions

#### Prerequisites

- Python 3.10+
- Git
- (Optional) GPU for faster inference

#### Steps

1. Clone the Repository:
```
>> https://github.com/hades-init/toxic-comment-classification.git
>> cd toxic-comment-classification
```

2. Install Dependencies:
Create and activate a virtual environment (use pip or conda):
```
>> conda create -p venv python=3.10
```

3. Install required packages:
```
>> pip install -r requirements.txt
```

4. Download data 
```
>> python -m src.data.make_dataset
```

### Running the App

#### 1. Start the Application:
Run the main entry point
```
>> python main.py
```

#### 2. Access the Web Interface:
Open your browser and navigate to: 
```
http://localhost:8000
```
Enter a comment in the text box and click "Submit" to see toxicity scores.


### Orchestration script (optional)

To run the entire pipeline - download and process dataset, train and evaluate:
```
>> python run_pipeline.py
```
This is not recommended as training take forever on a CPU. Recommed to use GPU or Kaggle notebooks. Each of the components of the pipeline is also an individual python script. 