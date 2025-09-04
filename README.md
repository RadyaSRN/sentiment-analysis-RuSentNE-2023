[![Open In nbviewer](https://img.shields.io/badge/Jupyter-nbviewer-orange?logo=jupyter)](
https://nbviewer.org/github/RadyaSRN/sentiment-analysis-RuSentNE-2023/blob/main/notebooks/sentiment_analysis.ipynb)
[![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](
https://www.kaggle.com/kernels/welcome?src=https://github.com/RadyaSRN/sentiment-analysis-RuSentNE-2023/blob/main/notebooks/sentiment_analysis.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/github/RadyaSRN/sentiment-analysis-RuSentNE-2023/blob/main/notebooks/sentiment_analysis.ipynb)
[![W&B Report](https://img.shields.io/badge/Weights%20&%20Biases-Report-orange?logo=weightsandbiases)](
https://wandb.ai/radyasrn-mipt/NLP-spring-2025/reports/sentiment-analysis-RuSentNE-2023--VmlldzoxMjc3ODUyMw)

# Sentiment Analysis for the data from the RuSentNE-2023
Sentiment analysis in relations to the named entities from the data from the **[RuSentNE-2023](https://github.com/dialogue-evaluation/RuSentNE-evaluation) competition** using the **[RuBERT](https://huggingface.co/DeepPavlov/rubert-base-cased) model**, it's modifications, and various techniques.

![Illustration](images/illustration.png)

### Sentiment analysis dataset
For model training the data from the [RuSentNE-2023](https://github.com/dialogue-evaluation/RuSentNE-evaluation) repository was used.

### Usage
* The first option is to open and run the notebook `/notebooks/sentiment_analysis.ipynb` with comments and visualizations in Kaggle or Google Colab.

* The second option is cloning the repo, installing the needed requirements, and working locally:
```
git clone https://github.com/RadyaSRN/sentiment-analysis-RuSentNE-2023.git
cd sentiment-analysis-RuSentNE-2023
conda create -n sentanalysis python=3.10
conda activate sentanalysis
pip install -r requirements.txt
```
