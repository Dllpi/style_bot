import os
import tensorflow as tf
from img.model import Artgan
from definitions import INPUT_IMAGES_DIR

tf.set_random_seed(228)


def parse_list(str_value):
    if ',' in str_value:
        str_value = str_value.split(',')
    else:
        str_value = [str_value]
    return str_value


def process_im(file_name, model_name='model_van-gogh'):
    tf.reset_default_graph()

    tfconfig = tf.ConfigProto(allow_soft_placement=False)
    tfconfig.gpu_options.allow_growth = True
    with tf.Session(config=tfconfig) as sess:
        model = Artgan(sess=sess, model_name=model_name, batch_size=1, image_size=640)

        model.inference(img_path=os.path.join(INPUT_IMAGES_DIR, file_name),
                        resize_to_original=False,
                        ckpt_nmbr=None)

        sess.close()
    return True