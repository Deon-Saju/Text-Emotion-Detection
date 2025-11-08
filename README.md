# Text-Emotion-Detection

### 📌 Overview
The goal of this study is to identify human emotions in brief social media writings, such as tweets.  A Transformer-based model (BERT) is refined to categorize emotions including joy, sadness, love, rage, fear, and surprise using Natural Language Processing (NLP) approaches.
 The experiment shows how sentiment and emotion recognition problems may be effectively and consistently handled using contemporary deep learning systems.

### 🎯 Objectives

- Fine-tune has a pre-trained Transformer model on an open-source emotion dataset.

- Accurately classify emotions in short text inputs.

- Build a reproducible workflow for NLP model training, evaluation, and deployment.

- Provide an interactive demo using Gradio .

### 🧩 Dataset

Dataset: Emotion Dataset from Hugging Face ( dair-ai/emotion)

Description: 20,000+ tweets labeled with six emotion categories:
anger, fear, joy, love, sadness,surprise

Split:

Train: 16,000 samples

Validation: 2,000 samples

Test: 2,000 samples


### ⚙️ Methodology
1. Approach

- Type: Supervised Learning

- Model: Fine-tuned bert-base-uncasedusing the Hugging Face Transformerslibrary.

- Frameworks: PyTorch + Hugging Face + Gradio

2. Preprocessing

- Tokenization using AutoTokenizer(max length = 128)

- Lowercasing and truncation

- Split into train/validation/test

3. Training

- Fine-tuned using the TrainerAPI

- Optimizer: AdamW

- Batch size: 16

- Epochs: 3

- Evaluation metric: Accuracy

4. Evaluation

- Evaluated on validation and test sets

- Metrics: Accuracy, F1-score, and confusion matrix

5. Deployment

- Final model saved locally in/emotion_model_saved/

- Deployed with Gradio for live testing


### 📊 Results

<img width="547" height="261" alt="image" src="https://github.com/user-attachments/assets/48f5a083-4daf-4b8c-a655-1645f38e3fdf" />

#### Confusion Matrix
<img width="574" height="590" alt="image" src="https://github.com/user-attachments/assets/d300ebc3-9bea-42ee-8981-2b35ee5638cd" />

#### Model prediction examples:
| Input                         | Predicted Emotion |
| ----------------------------- | ----------------- |
| “I’m feeling amazing today!”  | Joy               |
| “I hate when people lie.”     | Anger             |
| “You make me feel loved ❤️”   | Love              |
| “That movie scared me a lot!” | Fear              |



### Interface
A interactive was created using Gradio.
<img width="1910" height="862" alt="image" src="https://github.com/user-attachments/assets/419a3ae1-eb51-4f21-a1b5-7e4c546fc8bc" />


