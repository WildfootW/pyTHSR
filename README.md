# book THSR tickets with Restful Api

* There are 2 approach to handle annoying Captcha.
  1. Fetch _unlabeled_ Captcha from THSR server, use CycleGAN to transfer its weird style to clean one.
  2. Use **captcha generator** python package with some modification to match the real one. Get _labeled_ Captcha.

## Experiment

### Attempt1 -- Train on my generator, Valid on real captcha

Test with 1000 real hand-labeled THSR Captcha

> word accuracy: 2.1%
> char accuracy: 35.6%

Test with 1000 generated Captcha

> word accuracy: 89.4%
> char accuracy: 97.1%

### Attempt2 -- Train on 900 real captcha, Valid on 100 real captcha (random seed 666)

Test with 1000 real hand-labeled THSR Captcha

> word accuracy: 70.2%
> char accuracy: 90.45%

Test with 1000 generated Captcha

> word accuracy: 0.4%
> char accuracy: 19.175%

( exp\_result\_{1|2}.csv stores the experiment result, can be reproduced by `python -O experiment.py -1` )

## THSR Captcha Generator
```python
>>> # At root directory of pyTHSR
>>> from captcha.image import ImageCaptcha
>>> font = 'fonts/MyriadPro-Semibold.otf'
>>> generator = ImageCaptcha(width=WIDTH, height=HEIGHT, fonts=[font], font_sizes=[42,], curve_width=5)
>>> im = generator.create_THSR_captcha('Y2NK', isImg=True)
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

### Book ticket in 60 seconds
1. Edit `_secret.py`, fillin values of 'uid' and 'phone' ( and 'email' if needed )
```python
secret = [{
    'uid': 'A123456789',
    'phone': '0911111111',
    'email': 'xxx@gmail.com',
}, ]
```
2. Edit `book.py`, adjust lines below
```python
126     startDate = '2018/??/??'
127     backDate = '2018/??/??'
128     isStudent = True # ( or False )
129     config.includeBack = True #  including return from B to A
130     config.MAX_PASS = 100
131 
132     from _secret import secrets
133     users = list(map(lambda x: packUserInfo(**x), secrets))
134     userInfo = users[0]
135     # Adjust here to fit your needs
136     data = packInfo(toDate=startDate, toTime='20:30',
137             backDate=backDate, backTime='20:30',
138             from_='台南', to_='左營',   # available city name defined at line 58
139             tick_n=[0, 0, 0, 0, len(users)],
140             isStudent=isStudent, incBack=config.includeBack)
```

3. Run `python book.py`

## TaskList:
  - [x] Train a simple [OCR model](ocr_model.h5) with this[approach 2] dataset
  - [x] Train a simple [OCR model](pure.h5) on hand-labeled real dataset
  - [ ] Add **denoise constraint** when trains _ocr-model_
  - [ ] Refactor data/model process flow. ex. Encapsule into a class
  - [ ] Logging System
  - [ ] 5-character Captcha included ( updated 2018/03/07 )
