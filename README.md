### Contents

- [tf_aug_utils.py](tf_aug_utils.py) contains several functions for image augmentations. All are built using tensorflow and all are compatible with tensorflow Dataset pipelines for TPU training. All  functions can be `@tf.function` decorated. Augmentations include:
  - Random zoom-out with pan
  - Random zoom-in of random area
  - Random rotate
  - Random shear
  - Central zoom

