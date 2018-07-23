"""Author: Brandon Trabucco.
Calculate the model loss for reconstructing coordinates.
"""


import tensorflow as tf


from image_annotations.abstract import Abstract
from image_annotations.model.model_inputs import ModelInputs
from image_annotations.model.model_ops import ModelOps


class TrainLosses(Abstract):
    """Utility class to calulate model losses.
    """

    def __init__(self, input_file_pattern):
        """Initialize the input and train parameters.
        """

        self.input_file_pattern = input_file_pattern
        self.loss_fn = tf.nn.l2_loss

    
    def start(self):
        """Calculate a single update of the loss.
        """

        model_inputs = ModelInputs(self.input_file_pattern, True)
        video_ids, image_ids, xss, yss, images = model_inputs.start()
        model_ops = ModelOps(images, True, tf.shape(xss)[1])
        predicted_xss, predicted_yss = model_ops.start()
        
        xss_loss = self.loss_fn(predicted_xss - xss)
        yss_loss = self.loss_fn(predicted_yss - yss)
        tf.losses.add_loss(xss_loss)
        tf.losses.add_loss(yss_loss)
        total_loss = tf.losses.get_total_loss()

        return total_loss
        