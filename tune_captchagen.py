from collections import OrderedDict
from core import WIDTH, HEIGHT, captcha

text_space = OrderedDict([
    ('left_rate', []),
    ('width_rate', []),
    ('y_low_rate', []),
    ('y_up_rate', []), ])

# all integers
curve_space = OrderedDict([
    ('rad_lb', []), 
    ('rad_ub', []), 
    ('dx_lb', []), 
    ('dx_ub', []), 
    ('dy_lb', []), 
    ('dy_ub', []), 
    ])

noise_space = OrderedDict([
    ('lamb', []),
    ('std', []),
    ('fn', [])
    ])


if __name__ == '__main__':
    pass

