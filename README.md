# Sentiment Analysis Classifier (PT/BR)

Este é um projeto de **análise de sentimentos** que classifica textos como positivos ou negativos utilizando aprendizado de máquina. O modelo foi treinado em datasets rotulados e inclui pré-processamento avançado como limpeza de texto, stemming e lemmatization.

## 📂 Estrutura do Projeto
- `databases/` - Contém os datasets utilizados (`imdb`, `amazon`, `yelp`).
- `sentiment_analysis.py` - Script principal com a implementação do classificador.
- `requirements.txt` - Dependências do projeto.

## 🚀 Funcionalidades
- Pré-processamento do texto (remoção de URLs, stop words, stemming e lemmatization).
- Treinamento e teste de modelos com `TfidfVectorizer` e `Logistic Regression`.
- Geração de métricas de desempenho:
  - Relatório de classificação.
  - Matriz de confusão.
  
## 🧪 Exemplos de Uso
### Visualizar os primeiros registros do dataset:
```python
print(df.head())
```

### Treinar o modelo:
```python
model.fit(X_train_vectorized, y_train)
```

### Avaliar o modelo:
```python
print(classification_report(y_test, y_pred))
```

## ⚙️ Requisitos
- Python 3.11 ou superior
- Bibliotecas: 
  - pandas
  - numpy
  - scikit-learn
  - matplotlib

Instale as dependências com:
```bash
pip install -r requirements.txt
```


---

# Sentiment Analysis Classifier (ENG)

This is a **sentiment analysis** project that classifies text as positive or negative using machine learning. The model is trained on labeled datasets and includes advanced preprocessing techniques such as text cleaning, stemming, and lemmatization.

## 📂 Project Structure
- `databases/` - Contains the datasets used (`imdb`, `amazon`, `yelp`).
- `sentiment_analysis.py` - Main script with the classifier implementation.
- `requirements.txt` - Project dependencies.

## 🚀 Features
- Text preprocessing (removal of URLs, stop words, stemming, and lemmatization).
- Training and testing models with `TfidfVectorizer` and `Logistic Regression`.
- Performance metrics:
  - Classification report.
  - Confusion matrix.
  
## 🧪 Usage Examples
### View the first records of the dataset:
```python
print(df.head())
```

### Train the model:
```python
model.fit(X_train_vectorized, y_train)
```

### Evaluate the model:
```python
print(classification_report(y_test, y_pred))
```

## ⚙️ Requirements
- Python 3.11 or higher
- Libraries:
  - pandas
  - numpy
  - scikit-learn
  - matplotlib

Install the dependencies with:
```bash
pip install -r requirements.txt
```
