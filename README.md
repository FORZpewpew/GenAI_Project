# GenAI_Project
GenAI project that aims to use AI to generate funny captions for images 


# Changes from project check point

A custom encoder-decoder network using ResNet50 for image feature extraction and LSTM for caption generation took tens of hours to train, yet it still failed to produce even remotely reasonable captions. So instead we decided to use pre-trained model based on Generative Image-to-text Transformer (GIT). In contrast, the GIT model provided significantly better caption quality out of the box, drastically reducing both development and computational costs, thanks to being pre-trained on massive datasets.

# Training

image_captioning.ipynb contains code for training model and save it's weights to weights folder

# How to run captioning app
Before running you should train model and save weights to weights folder
```
pip install -r requirements.txt

streamlit run app.py
```