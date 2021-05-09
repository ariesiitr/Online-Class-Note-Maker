# Online-Class-Note-Maker
Recruitment Project for 2nd year.

System that generates summarised text for the text avialble in literary form of study materials like pdf, etc along with natural language captions for any image.

## Dataset
The image captioning model is trained on [Flickr8k Dataset](https://illinois.edu/fb/sec/1713398)

## Model
<div align="center">
  <img src="model.png"><br><br>
</div>

## Performance
The model has been trained for 20 epoches on 6000 training samples of Flickr8k Dataset.

----------------------------------

## Requirements
- Python 3.6
- sumy 
- pdfplumber
- fitz
- tensorflow
- pillow
- matplotlib
- h5py
- keras
- numpy
- pydot
- nltk
- progressbar2
- pytesseract
- PyPDF2
- io

These requirements can be easily installed by:
  `pip install -r requirements.txt`


## Scripts

- __caption_generator.py__: The base script that contains functions for model creation, batch data generator etc.
- __prepare_data.py__: Extracts features from images using VGG16 imagenet model. Also prepares annotation for training. Changes have to be done to this script if new dataset is to be used.
- __train_model.py__: Module for training the caption generator.
- __eval_model.py__: Contains module for evaluating and testing the performance of the caption generator, currently, it contains the [BLEU](https://en.wikipedia.org/wiki/BLEU) metric.

## Usage

### Pre-trained model
1. Download pre-trained weights from [releases](https://github.com/Div99/Image-Captioning/releases)
2. Move `model_weight.h5` to `models` directory
3. Prepare data using `python prepare_data.py`
4. For inference on example image, run: `python eval_model.py -i [img-path]`

### From scratch
After the requirements have been installed, the process from training to testing is fairly easy. The commands to run:
1. `python prepare_data.py`
2. `python train_model.py`
3. `python eval_model.py`

After training, evaluation on an example image can be done by running:  
`python eval_model.py -m [model-checkpoint] -i [img-path]`
