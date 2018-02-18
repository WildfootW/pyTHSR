## book THSR tickets with Restful Api

* There are 2 approach to handle annoying Captcha.
  1. Fetch _unlabeled_ Captcha from THSR server, use CycleGAN to transfer its weird style to clean one.
  2. Use **captcha generator** python package with some modification to match the real one. Get _labeled_ Captcha.

# THSR Captcha Generator
```python
>>> # At root directory of pyTHSR
>>> from captcha.image import ImageCaptcha
>>> font = 'fonts/MyriadPro-Semibold.otf'
>>> generator = ImageCaptcha(width=WIDTH, height=HEIGHT, fonts=[font], font_sizes=[42,])
>>> im = generator.create_THSR_captcha('Y2NK', color='black', background='#fff', pen_size=5, isImg=True)
```
![sample](sample_Y2NK.bmp)

* TODO:
  * train a simple OCR model with this[approach 2] dataset
