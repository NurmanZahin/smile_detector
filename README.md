[![LinkedIn][linkedin-shield]][linkedin-url]

# Smile Detector
> A simple smile detector using pretrained ResNet and deployed on using flask.



## Table of Contents
- [Introduction](#introduction)
- [Getting Started](#getting-started)
  - [Clone](#clone)
  - [Installing dependencies](#installing-dependencies)
- [Usage](#usage)
- [Acknowledgements](#acknowledgements)
  - [Articles](#articles)
  - [Codes that were used](#codes-that-were-used)
  - [Dataset](#dataset)
- [License](#license)


## Introduction
This is a simple image classification project when I had first learnt about computer vision. I had decided to make it 
into a simple web application using the flask.


## Getting Started

### Clone
- Cloning repo to your local setup
```shell
$ git clone https://github.com/NurmanZahin/smile_detector.git
```

### Installing dependencies
- Create a new conda environment 
- Install dependencies using the environment.yml
```shell
$ conda env create -f environment.yml
$ conda activate smile-detector 
```
### Installing dependencies
Download the Olivetti dataset from [here](https://www.kaggle.com/sahilyagnik/olivetti-faces). Go through the notebooks 
to create the necessary folders and to train your model. With the finetuned model ready, you can now run the web app or
do inference on your images.

## Usage
To run the web application 
```shell
$ python -m src.app path/to/input_image
```

To run the smile detector
```shell
$ python -m src.inference path/to/input_image
```


## Acknowledgements
### Dataset
- Olivetti Faces Dataset (AT&T Laboratories Cambridge)


## License

[![MIT License][mit-license-shield]][mit-license-url]

- **[MIT license](http://opensource.org/licenses/mit-license.php)**



[mit-license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=flat-square
[mit-license-url]: https://badges.mit-license.org/
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/nurman-jupri-20655814a
[product-screenshot]: images/screenshot.png
[article1-url]: https://towardsdatascience.com/beginners-guide-to-building-a-singlish-ai-chatbot-7ecff8255ee
[article2-url]: https://towardsdatascience.com/generating-singlish-text-messages-with-a-lstm-network-7d0fdc4593b6
