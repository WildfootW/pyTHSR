# book THSR tickets with Restful Api

* There are 2 approach to handle annoying Captcha.
  1. Fetch _unlabeled_ Captcha from THSR server, use CycleGAN to transfer its weird style to clean one.
  2. Use **captcha generator** python package with some modification to match the real one. Get _labeled_ Captcha.

## Experiment

Test with 1000 real hand-labeled THSR Captcha

> word accuracy: 2.1%
> char accuracy: 35.6%

Test with 1000 generated Captcha

> word accuracy: 89.4%
> char accuracy: 97.1%

( exp\_result.csv stores the experiment result, can be reproduced by `python experiment.py` )

## THSR Captcha Generator
```python
>>> # At root directory of pyTHSR
>>> from captcha.image import ImageCaptcha
>>> font = 'fonts/MyriadPro-Semibold.otf'
>>> generator = ImageCaptcha(width=WIDTH, height=HEIGHT, fonts=[font], font_sizes=[42,])
>>> im = generator.create_THSR_captcha('Y2NK', color='black', background='#fff', pen_size=5, isImg=True)
```
![sample](sample_Y2NK.bmp)

## Get Started
### Do prediction on random sampled images
```bash
$ python simple_cnn.py --load
```

### Train from pretrained denoise model
```bash
$ python simple_cnn.py --load --train_ocr
```


## TODO:
  - [x] Train a simple [OCR model](ocr_model.h5) with this[approach 2] dataset
  - [ ] Add **denoise constraint** when trains _ocr-model_
