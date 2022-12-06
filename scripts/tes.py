
import tensorflow as tf
import numpy as np
import cv2


def foo(a, b):
    if (a>b):
        return a
    return b

print(foo(4, 4))

def categorical(image, mask, is_categorical=False):
    if not is_categorical:
        label = tf.keras.utils.to_categorical(mask, num_classes = 4).astype(np.uint8)
    
    image = cv2.normalize(image[:, :, :, 0], None, alpha=0, beta=255,
                            norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).astype(np.uint8)
                            
    labeled_image = np.zeros_like(label[:, :, :, 1:])
    labeled_image[:, :, :, 0] = image * (label[:, :, :, 0])
    labeled_image[:, :, :, 1] = image * (label[:, :, :, 0])
    labeled_image[:, :, :, 2] = image * (label[:, :, :, 0])

    # color labels
    labeled_image += label[:, :, :, 1:] * 255
    return labeled_image
        
  
image_input = np.random.rand(240, 240, 155, 4)
true_label = np.random.rand(240, 240, 155)

mask = categorical(image_input, true_label)
print(mask.shape)
