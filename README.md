# Fine-Tuning SpeechT5 for Tamazight Text-to-Speech

Companion Repository for: “Fine-Tuning SpeechT5 for Tamazight Text-to-Speech: A Foundation for Educational Technology in Low-Resource Languages” (Submitted to International Journal of Speech Technology)

## Overview

This repository contains the resources, scripts, and configuration files used to fine-tune Microsoft’s SpeechT5 model for text-to-speech (TTS) synthesis in Tamazight, a low-resource language. The training pipeline uses Hugging Face Transformers and Datasets libraries, with detailed preprocessing, model training, and evaluation steps made publicly available.

## Quick Links

- Preprocessed Training Set (HF-compatible):  
  https://drive.google.com/file/d/155puJitrHIbz6-8NUUohmXR2OJu8m2uj/view?usp=sharing

- Preprocessed Validation Set (HF-compatible):  
  https://drive.google.com/file/d/1G8AoOitHYw4QDEmWMeZlRO73WTdxA9el/view?usp=sharing

- Raw Audio Archive (with CSV files for train, val, and test):  
  https://drive.google.com/file/d/1i7HA2NAwSvPFzdKtPTeMPbvcilmKYq_8/view?usp=drive_link

- Kaggle Preprocessing Notebook:  
  https://www.kaggle.com/code/youssefamk/speecht5-preprocessing

- Training Pipeline (Colab):  
  https://colab.research.google.com/drive/1ZyoRbCt15QcJL1Yg-FSX7BeLlE_w407y

- Checkpoints + Logs + Best models:  
  https://drive.google.com/drive/folders/1LaadQjoFgf9axoVAI5VbGg4pFfVN9cJH?usp=sharing

## Dataset Sources

This project was built using the following open-source speech datasets:

| Dataset              | Language  | License     | Link                                                               |
|----------------------|-----------|-------------|--------------------------------------------------------------------|
| Common Voice v13     | Tamazight | CC-0        | https://commonvoice.mozilla.org                                    |
| tamazight_asr        | Tamazight | CC-BY       | https://huggingface.co/datasets/TutlaytAI/tamazight_asr            | 
| CSS10-Tamazight      | Tamazight | CC-BY       | https://huggingface.co/datasets/Tamazight-NLP/tamawalt-n-imZZyann  | 

These datasets were cleaned, transliterated from Tifinagh to Latin, and filtered during preprocessing. Manifest files describing splits (train, val, test) are included with the data archive linked above.

## How to Cite

If you use this repository or its resources in your work, please cite the corresponding article (preprint link will be shared upon publication).

## Contact

For questions, corrections, or collaborations, please open an issue or reach out via the GitHub repository.
