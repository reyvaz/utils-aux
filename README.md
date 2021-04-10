### Contents

- [html_metrics.py](html_metrics.py) Contains functions to create confusion matrices and binary classification reports using HTML.<br><br>
  I was inspired to create these simply by aesthetics. For some time, I have been using “heated map” confusion matrices as an excuse to add color into otherwise duller notebooks and reports. However, options such as matplotlib and seaborn tended to blurry the text in the plots. This was particularly true when the plots are saved as jpg or png images and then (often automatically) resized to fit different types of displays. <br><br>
  My way around this was to create these functions, where text is displayed directly into the “graphics” and never converted to image. These can then be displayed in Jupyter notebooks, webpages, or even Microsoft documents. 

  [![Colab Demo](https://img.shields.io/badge/Colab%20Demo-grey?style=for-the-badge&logo=Google-Colab)](https://githubtocolab.com/reyvaz/utils-aux/blob/master/html_metrics_demo.ipynb) 


- [tf_aug_utils.py](tf_aug_utils.py) contains several functions for image augmentations. All are built using tensorflow and all are compatible with tensorflow Dataset pipelines for TPU training. All  functions can be `@tf.function` decorated.  Augmentations include: <br>[**Update:** These functions have been updated and integrated into the [tpu_segmentation](https://github.com/reyvaz/tpu_segmentation/blob/master/augmentations.py) library. The updated versions support augmenting images or images + masks and all are compatible with Tensorflow Dataset pipeplines for TPU training].

  - Random zoom-out with random pan
  - Random zoom-in of random area
  - Random rotate
  - Random shear
  - Central zoom


- [notebook_utils.py](notebook_utils.py) contains various time utils; classification reporting; confusion matrix plot.<br>[**Update:** Updated versions of the confusion matrices and binary classification reports are in [html_metrics.py](html_metrics.py)].