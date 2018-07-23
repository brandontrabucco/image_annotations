"""Author: Brandon Trabucco.
Utility class for implementing a CNN model for regressing joint positions.
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
from tensorflow.contrib.slim.python.slim.nets.inception_v3 import inception_v3_base
slim = tf.contrib.slim


from image_annotations.abstract import Abstract


class ModelOps(Abstract):
    """Utility class for CNN operations.
    """

    def __init__(self, images, is_training, num_points):
        """Create some network configuration params.
        """

        self.images = images
        self.is_training = is_training
        self.num_points = num_points
        self.weight_decay = 0.00004
        self.stddev = 0.1
        self.dropout_keep_prob = 0.8
        self.use_batch_norm = True
        self.batch_norm_params = None
        self.add_summaries = True
        self.scope = "InceptionV3"



    def start(self):
        """Run the CNN and retrieve outputs from the model.
        Returns:
            xs: tf.float32 Tensor of x positions
            ys: tf.float32 Tensor of y positions
        """

        if self.use_batch_norm:
            # Default parameters for batch normalization.
            if not self.batch_norm_params:
                self.batch_norm_params = {
                    "is_training": self.is_training,
                    "trainable": True,
                    # Decay for the moving averages.
                    "decay": 0.9997,
                    # Epsilon to prevent 0s in variance.
                    "epsilon": 0.001,
                    # Collection containing the moving mean and moving variance.
                    "variables_collections": {
                        "beta": None,
                        "gamma": None,
                        "moving_mean": ["moving_vars"],
                        "moving_variance": ["moving_vars"],
                    }
                }
        else:
            self.batch_norm_params = None

        weights_regularizer = tf.contrib.layers.l2_regularizer(self.weight_decay)

        with tf.variable_scope(self.scope, "InceptionV3", [self.images]) as scope:
            with slim.arg_scope(
                    [slim.conv2d, slim.fully_connected],
                    weights_regularizer=weights_regularizer,
                    trainable=True):
                with slim.arg_scope(
                        [slim.conv2d],
                        weights_initializer=tf.truncated_normal_initializer(stddev=self.stddev),
                        activation_fn=tf.nn.relu,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=self.batch_norm_params):
                    net, end_points = inception_v3_base(self.images, scope=scope)

        # Add summaries.
        if self.add_summaries:
            for v in end_points.values():
                tf.contrib.layers.summaries.summarize_activation(v)

        with tf.variable_scope(self.scope, "Logits", [net]) as scope:
            net = tf.reduce_mean(net, axis=[1, 2])
            xs = slim.fully_connected(net, self.num_points, scope=scope)
            ys = slim.fully_connected(net, self.num_points, scope=scope)

        return xs, ys