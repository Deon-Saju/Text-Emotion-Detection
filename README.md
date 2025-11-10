#  “Emotion Detection from Tweets using Transformer Fine-Tuning — a project demonstrating NLP model adaptation and deployment with Gradio.”


## 📌 Overview
The goal of this study is to identify human emotions in brief social media writings, such as tweets.  A Transformer-based model (BERT) is refined to categorize emotions including joy, sadness, love, rage, fear, and surprise using Natural Language Processing (NLP) approaches.
 The experiment shows how sentiment and emotion recognition problems may be effectively and consistently handled using contemporary deep learning systems.

## 🎯 Objectives

- Fine-tune has a pre-trained Transformer model on an open-source emotion dataset.

- Accurately classify emotions in short text inputs.

- Build a reproducible workflow for NLP model training, evaluation, and deployment.

- Provide an interactive demo using Gradio .

## 🧩 Dataset

Dataset: Emotion Dataset from Hugging Face ( dair-ai/emotion)

Description: 20,000+ tweets labeled with six emotion categories:
anger, fear, joy, love, sadness,surprise

Split:

Train: 16,000 samples

Validation: 2,000 samples

Test: 2,000 samples


## ⚙️ Methodology
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


## 📊 Results

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



## Quick Start
1. Clone the repository
   ```
   git clone https://github.com/yourusername/emotion-detection.git
   cd emotion-detection
   ```

2. Install dependencies
   ```
   pip install -r requirements.txt
   ```

3. Run training
   ```
   !python fine_tuning.py
   ```
   
4. Run 
   ```
   !python gradio_app.py
   ```
   

## 🔗 Model Access
The fine-tuned model is saved in Google Drive: https://drive.google.com/drive/folders/1UcSYntDESf5sG_xwHlaXGBXeb8rNgvIx?usp=drive_link

`/content/drive/MyDrive/emotion_model_saved`

### The folder usually contains 

- model.safetensors
- config.json
- special_tokens_map.json
- tokenizer_config.json
- tokenizer.json
- training_args.bin
- vocab.txt


To run the app:
1. Mount Google Drive in your Colab session.
2. Make sure the folder `emotion_model_saved` exists in your Drive.
3. Run `gradio_app.py` — it’ll automatically load from Drive.


## Interface
A interactive was created using Gradio.
<img width="1910" height="862" alt="image" src="https://github.com/user-attachments/assets/419a3ae1-eb51-4f21-a1b5-7e4c546fc8bc" />


## 🧩 Reproducibility

To reproduce the results:
1. Clone this repository.
2. Open `fine_tuning.py` in Google Colab.
3. Run all cells to fine-tune the model and save it.
4. Open `gradio_app.py` to launch the web interface.
5. Test your model with any text!


## Future Work
- Fine-tune with larger emotion datasets.
- Add multilingual emotion support (French, Spanish, etc.).
- Deploy the app with Streamlit or Hugging Face Spaces.
- Integrate emoji-based sentiment detection.


## 📚 References

- [Hugging Face Datasets: Emotion](https://huggingface.co/datasets/dair-ai/emotion)
- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [Gradio](https://www.gradio.app/)


## 👨‍🎓 Author
Academic Project: M2 Deep Learning with Python Project
Date: November 2025

## 📜 License
This project is for academic purposes. 
