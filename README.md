### Contents

- [tf_aug_utils.py](tf_aug_utils.py) contains several functions for image augmentations. All are built using tensorflow and all are compatible with tensorflow Dataset pipelines for TPU training. All  functions can be `@tf.function` decorated.  Augmentations include: <br>(**P.S.** Also look [here](https://github.com/reyvaz/tpu_segmentation/blob/master/augmentations.py) for updated functions integrated into the image segmentation repo).

  - Random zoom-out with random pan
  - Random zoom-in of random area
  - Random rotate
  - Random shear
  - Central zoom

- [notebook_utils.py](notebook_utils.py) contains various time utils; classification reporting; confusion matrix plot.