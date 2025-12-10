## Sentiment Analysis of Tweets

This project considers the multiclass classification problem of sentiment analysis for Tweets. The goal of this project is to develop various deep learning models to classify Tweets based on their text content using natural language processing (NLP) techniques. The dataset for this project was sourced from [Kaggle](https://www.kaggle.com/datasets/yessicatuteja/sentiment-analysis-of-tweets), containing 40,000 obvservations of Tweets labeled among 13 sentiment classes. At a larger scale, similar approaches as this project can be adapted for applications in social media moderation and hate speech detection.

This project explored many different deep learning architectures, focusing on model types that are suitable for sequential data including:

- Simple Recurrent Neural Networks (RNNs)
- Gated RNNs (GRU, LSTM)
- Bidirectional RNNs
- Transformer Models

These deep learning models were compared against baseline models from traditional machine learning techniques, such as gradient boosted trees (XGBoost) and multilayer perceptrons (MLPs). The best deep learning approach was the Bidirectional RNN model which achieved an accuracy of **36.8%** on the testing set, a small improvement over the MLP baseline of **36.1%**. The inherent difficulty of this classification problem limited the performance of all model types, and the MLP remained one of the top-performing models, but these results illustrate the potential of applying a deep learning approach for this task.

This was completed as a final project for STAT 362 at Northwestern University in Fall 2025.

## Repo Structure

- [`code/`](code/): Jupyter notebooks with analysis code
- [`colab/`](colab/): Supplementary files for running notebooks in Google Colab
- [`data/`](data/): Dataset files
- [`exploration/`](exploration/): Jupyter notebooks for initial exploration and diagnostic checks 
- [`reports/`](reports/): Course submission files

## Instructions to Run Code

**Note:** Code was originally run using the T4 GPU (High-RAM) in Google Colab, and the results were saved locally. All notebooks can be run independently in any order.

### Running Code Locally

This is **not recommended** without GPU compatibility due to high computational costs.

1. Create a Poetry environment using the included `pyproject.toml` file (i.e. run `poetry install` in the Terminal)
2. Open a notebook within the `code/` directory (i.e. `FINAL_baseline.ipynb`)
3. Run the notebook cells from start to finish

### Running Code in Google Colab

1. Import a notebook from the `code/` directory into Google Colab (i.e. `FINAL_baseline.ipynb`)
2. Connect to a hosted runtime (i.e. High-RAM T4 GPU)
3. Upload the files in the `colab/` directory to the Colab environment (i.e. `Sentiment_Analysis.csv`, `util_funcs.py`)
4. Adjust the file paths to match the new directory structure
   a. `"../data/Sentiment_Analysis.csv"` will likely be replaced by `"Sentiment_Analysis.csv"` for loading data
5. Run the notebook cells from start to finish
