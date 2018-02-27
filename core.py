from functools import partial

from captcha.image import ImageCaptcha

WIDTH, HEIGHT, CHAR_NUM = 128, 48, 4
classes = u'ACFHKMNQPRTYZ234579'
class_num = len(classes)

font = 'fonts/MyriadPro-Semibold.otf'
captcha = partial(ImageCaptcha, width=WIDTH, height=HEIGHT, fonts=[font], font_sizes=[44,])

