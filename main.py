from models import *
from layer_utils import *
import os, sys

output_folder = 'output'
WIDTH, HEIGHT = 124, 48

cycle_gan = CycleGAN(shape=(HEIGHT, WIDTH, 1), bch_img_num=64, ps=256,
	task_name='denoise', pic_dir=output_folder)
try:
	cycle_gan.fit(epoch_num=200, disc_iter=5, save_period=1)
	cycle_gan.save(path='attempt', with_img=False, show_shapes=True)
except KeyboardInterrupt:
	K.clear_session()
	try:
		sys.exit(0)
	except SystemExit:
		os._exit(0)
else:
	K.clear_session()
