import math, re
import numpy as np
import tensorflow as tf

# Note all functions can be decorated with @tf.functions
# all have been tested in pipelines involving TPUs

# @tf.function
def random_zoom_out_and_pan(image, image_size, range = (0.5, 0.9)):
    '''
    Applies random zoom out with random pan to an image.
    Args:
        image: an image array
        image_size: a tuple (H, W, C)
        range: a tuple of floats (F1, F2) where F1<F2 and both are between 0
            and 1. The range of zoom-out factor.
    Returns:
        an image with random zoom-out + pan applied
    '''
    # Determine the random zoomed-in size
    random_factor = K.random_uniform((1,), range[0], range[1])
    height_width = tf.constant(image_size[:2])
    dim_deltas = tf.cast(height_width, dtype=tf.float32)*(1-random_factor)
    dim_deltas = tf.cast(dim_deltas, dtype = tf.int32)
    zoomed_size = height_width - dim_deltas

    # Determine the random padding for each side to create random pan
    paddings_c = tf.constant([0,0])
    pad_top = K.random_uniform((), 0, dim_deltas[0], dtype = tf.int32)
    pad_left = K.random_uniform((), 0, dim_deltas[1], dtype = tf.int32)
    pad_bottom = dim_deltas[0] - pad_top
    pad_right = dim_deltas[1] - pad_left
    aug_paddings = tf.stack([[pad_top,pad_bottom], [pad_left, pad_right], paddings_c], axis=0)

    # apply zoom-out and panning to image and mask, then reshape.
    zoomed_img = tf.image.resize(image, zoomed_size)
    zoomed_img = tf.pad(zoomed_img, aug_paddings)
    zoomed_img = tf.reshape(zoomed_img, image_size)
    return zoomed_img

# @tf.function
def random_zoom_in(image, image_size, range = (0.5, 0.9)):
    '''
    Applies random zoom-in of a random area in the image.
    Args:
        image: an image array
        image_size: a tuple (H, W, C)
        range: a tuple of floats (F1, F2) where F1<F2 and both are between 0
            and 1. The range of the zoom-in factor. i.e. a zoom-in factor of 0.9
            will randomly zoom-in an area of H*0.9 x W*0.9 of the original image

    Returns:
        an image with random zoom-out + pan applied
    '''
    h = image_size[0]
    w = image_size[1]
    aspect_ratio = w//h

    # Determine the random height and width of the area to be zoomed-in.
    random_factor = K.random_uniform((), range[0], range[1])
    dh = tf.cast(h*random_factor, dtype = tf.int32)
    dw = dh*aspect_ratio

    # Determine the random position of the random area within the original image
    max_x = w - dw
    max_y = h - dh
    x = K.random_uniform((), 0, max_x, dtype = tf.int32)
    y = K.random_uniform((), 0, max_y, dtype = tf.int32)

    # Crop + Resize -> Zoom-in
    cropped_img = tf.image.crop_to_bounding_box(image, y, x, dh, dw)
    zoomed_img = tf.image.resize(cropped_img, size = img_size)
    zoomed_img = tf.reshape(zoomed_img, image_size)
    return zoomed_img

# @tf.function
def rotate_img(image, DIM, rotate_factor = 45.0):
    # image - is one squared image of size [DIM,DIM,3]
    # output - image randomly rotated
    XDIM = DIM%2 #fix for odd size

    rot = rotate_factor * tf.random.normal([1], dtype='float32')
    rotation = math.pi * rot / 180. # degrees to radians

    def get_3x3_mat(lst):
        return tf.reshape(tf.concat([lst],axis=0), [3,3])

    c1 = tf.math.cos(rotation)
    s1 = tf.math.sin(rotation)
    one = tf.constant([1],dtype='float32')
    zero = tf.constant([0],dtype='float32')
    m = get_3x3_mat([c1,   s1,   zero,
                     -s1,  c1,   zero,
                     zero, zero, one])

    # LIST DESTINATION PIXEL INDICES
    x = tf.repeat(tf.range(DIM//2, -DIM//2,-1), DIM)
    y = tf.tile(tf.range(-DIM//2, DIM//2), [DIM])
    z = tf.ones([DIM*DIM], dtype='int32')
    idx = tf.stack( [x,y,z] )
    # ROTATE DESTINATION PIXELS ONTO ORIGIN PIXELS
    idx2 = K.dot(m, tf.cast(idx, dtype='float32'))
    idx2 = K.cast(idx2, dtype='int32')
    idx2 = K.clip(idx2, -DIM//2+XDIM+1, DIM//2)
    # FIND ORIGIN PIXEL VALUES
    idx3 = tf.stack([DIM//2-idx2[0,], DIM//2-1+idx2[1,]])
    d = tf.gather_nd(image, tf.transpose(idx3))
    return tf.reshape(d,[DIM, DIM,3])

# @tf.function
def shear_img(image, DIM, shear_factor = 7.0):
    # image - is one squared image of size [DIM,DIM,3]
    # output - randomly sheared image
    XDIM = DIM%2 #fix for odd size

    shr = shear_factor * tf.random.normal([1], dtype='float32')
    shear    = math.pi * shr    / 180. # degrees to radians

    def get_3x3_mat(lst):
        return tf.reshape(tf.concat([lst],axis=0), [3,3])
    # SHEAR MATRIX
    c2 = tf.math.cos(shear)
    s2 = tf.math.sin(shear)
    one  = tf.constant([1],dtype='float32')
    zero = tf.constant([0],dtype='float32')
    m = get_3x3_mat([one,  s2,   zero,
                     zero, c2,   zero,
                     zero, zero, one])
    # LIST DESTINATION PIXEL INDICES
    x   = tf.repeat(tf.range(DIM//2, -DIM//2,-1), DIM)
    y   = tf.tile(tf.range(-DIM//2, DIM//2), [DIM])
    z   = tf.ones([DIM*DIM], dtype='int32')
    idx = tf.stack( [x,y,z] )
    # ROTATE DESTINATION PIXELS ONTO ORIGIN PIXELS
    idx2 = K.dot(m, tf.cast(idx, dtype='float32'))
    idx2 = K.cast(idx2, dtype='int32')
    idx2 = K.clip(idx2, -DIM//2+XDIM+1, DIM//2)
    # FIND ORIGIN PIXEL VALUES
    idx3 = tf.stack([DIM//2-idx2[0,], DIM//2-1+idx2[1,]])
    d    = tf.gather_nd(image, tf.transpose(idx3))

    return tf.reshape(d,[DIM, DIM,3])

# @tf.function
def central_zoom(image, image_size, zoom_factor = 0.6):
    image = tf.image.central_crop(image, zoom_factor)
    image = tf.image.resize(image, image_size[:2])
    image = tf.reshape(image, image_size)
    return image
