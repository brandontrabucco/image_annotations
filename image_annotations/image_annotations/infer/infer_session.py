"""Author: Brandon Trabucco.
Calculate the model inference with respect to input images.
"""


import tensorflow as tf


from image_annotations.abstract import Abstract
from image_annotations.model.model_ops import ModelOps


class InferSession(Abstract):
    """Utility class to calulate model inference.
    """

    def __init__(self, train_dir, image_feed, num_points):
        """Initialize the input and train parameters.
        """

        self.train_dir = train_dir
        self.image_feed = image_feed
        self.num_points = num_points

    
    def start(self):
        """Calculate a single update of the loss.
        """

        # Build the TensorFlow graph.
        g = tf.Graph()
        with g.as_default():

            images = tf.constant(self.image_feed, dtype=tf.float32)
            while len(images.shape) < 4:
                images = tf.expand_dims(images, axis=0)
            images = tf.image.resize_image_with_crop_or_pad(
                images, 299, 299)

            model_ops = ModelOps(images, False, self.num_points)
            xss, yss = model_ops.start()

            # Set up the Saver for restoring model checkpoints.
            saver = tf.train.Saver()

            with tf.Session() as sess:
                saver.restore(tf.train.latest_checkpoint(self.train_dir))
                actual_xss, actual_yss = sess.run([xss, yss])

        return actual_xss, actual_yss
        