"""Author: Brandon Trabucco.
initiate a model training session.
"""


import tensorflow as tf


from image_annotations.abstract import Abstract
from image_annotations.train.train_losses import TrainLosses


class TrainSession(Abstract):
    """Utility class to calulate model updates.
    """

    def __init__(self, input_file_pattern, inception_checkpoint_file, 
            number_of_steps, batch_size, train_dir):
        """Initialize the input and train parameters.
        """

        self.input_file_pattern = input_file_pattern
        self.number_of_steps = number_of_steps
        self.batch_size = batch_size
        self.inception_checkpoint_file = inception_checkpoint_file
        self.train_dir = train_dir
        self.num_examples_per_epoch = 513
        self.optimizer = "SGD"
        self.initial_learning_rate = 2.0
        self.learning_rate_decay_factor = 0.5
        self.num_epochs_per_decay = 8.0
        self.clip_gradients = 5.0
        self.max_checkpoints_to_keep = 5
        self.log_every_n_steps = 100

    
    def start(self):
        """Calculate multiple gradient descent updates.
        """

        # Create training directory.
        if not tf.gfile.IsDirectory(self.train_dir):
            tf.logging.info("Creating training directory: %s", self.train_dir)
        tf.gfile.MakeDirs(self.train_dir)

        # Build the TensorFlow graph.
        g = tf.Graph()
        with g.as_default():
            # Build the model.
            train_losses = TrainLosses(self.input_file_pattern)
            total_loss = train_losses.start()

            # Setup the global step tensor
            global_step = tf.Variable(
                initial_value=0,
                name="global_step",
                trainable=False,
                collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

            # Set up the learning rate.
            learning_rate_decay_fn = None
            learning_rate = tf.constant(self.initial_learning_rate)
            if self.learning_rate_decay_factor > 0:
                num_batches_per_epoch = (self.num_examples_per_epoch /
                    self.batch_size)
                decay_steps = int(num_batches_per_epoch *
                    self.num_epochs_per_decay)

                def _learning_rate_decay_fn(learning_rate, global_step):
                    return tf.train.exponential_decay(
                        learning_rate,
                        global_step,
                        decay_steps=decay_steps,
                        decay_rate=self.learning_rate_decay_factor,
                        staircase=True)

                learning_rate_decay_fn = _learning_rate_decay_fn

            # Set up the training ops.
            train_op = tf.contrib.layers.optimize_loss(
                loss=total_loss,
                global_step=global_step,
                learning_rate=learning_rate,
                optimizer=self.optimizer,
                clip_gradients=self.clip_gradients,
                learning_rate_decay_fn=learning_rate_decay_fn)

            # Set up the Saver for saving and restoring model checkpoints.
            saver = tf.train.Saver(max_to_keep=self.max_checkpoints_to_keep)

            # Prepare to initialize with pretrained weights
            inception_variables = tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES, scope="InceptionV3")
            inception_saver = tf.train.Saver(inception_variables)

            def restore_fn(sess):
                tf.logging.info("Restoring Inception variables from checkpoint file %s",
                    self.inception_checkpoint_file)
                inception_saver.restore(sess, self.inception_checkpoint_file)

        # Run training.
        tf.contrib.slim.learning.train(
            train_op,
            self.train_dir,
            log_every_n_steps=self.log_every_n_steps,
            graph=g,
            global_step=global_step,
            number_of_steps=self.number_of_steps,
            init_fn=restore_fn,
            saver=saver)

        return True