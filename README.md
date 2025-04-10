# GenAI_Project
GenAI project that aims to use AI to generate funny captions for images 


# Progress so far
We found a labeled memes dataset and removed all captions from its images. Firstly, we tried using tesseract, but it couldn’t give a good enough result, so instead we applied keras_ocr.tools to detect text on images and cv2.inpaint to inpaint detected text, which got us a good enough image.

Now all that's left is to find the best model for generating captions.
For now the best performing model (that at least generated something) is a custom encoder-decoder network, where ResNet50 is used as encoder and LSTM as decoder. \
ResNet50 serves as the image feature extractor, encoding visual content into a format suitable for captioning, while the LSTM generates textual outputs. An attention mechanism was incorporated to allow the model to focus on significant regions of the image. However, this approach isn't definetely the best one, since it requires a vast dataset and extended training time, yet still struggling to produce contextually appropriate and humorous captions.


# Training

image_captioning.ipynb contains code for training model and save it's weights to weights folder

# How to run captioning app
Before running you should train model and save weights to weights folder
```
pip install -r requirements.txt

streamlit run app.py
```