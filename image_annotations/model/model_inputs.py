"""Author: Brandon Trabucco.
Loads training batches from the serialized tensorfloe dataset.
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf


from image_annotations.abstract import Abstract


class ModelInputs(Abstract):
    """Utility class to load training batches.
    """

    def __init__(self, input_file_pattern, is_training):
        """Initialize useful global variables.
        """

        self.reader = tf.TFRecordReader()
        self.input_file_pattern = input_file_pattern
        self.batch_size = 32
        self.values_per_input_shard = 16
        self.input_queue_capacity_factor = 16
        self.num_input_reader_threads = 4
        self.num_preprocess_threads = 4

        self.video_id_feature = "image/video_id"
        self.image_id_feature = "image/image_id"
        self.xs_feature = "image/xs"
        self.ys_feature = "image/ys"
        self.image_feature = "image/image"
        self.shape_feature = "image/shape"


    def parse_sequence_example(self, serialized):
        """Parses a tensorflow.SequenceExample into an image and caption.
        Args:
            serialized: A scalar string Tensor; a single serialized SequenceExample.
        Returns:
            video_id: tf.int64 scalar identifying the source video.
            image_id: tf.int64 scalar identifying the source frame.
            xs: tf.float32[] list of object points.
            ys: tf.float32[] list of object points.
            image: tf.float32[] list of flattened image pixels.
            shape: tf.int64[] list of the original image shape.
        """
        context, sequence = tf.parse_single_sequence_example(
            serialized,
            context_features={
                self.video_id_feature: tf.FixedLenFeature([], dtype=tf.int64),
                self.image_id_feature: tf.FixedLenFeature([], dtype=tf.int64)
            },
            sequence_features={
                self.xs_feature: tf.FixedLenSequenceFeature([], dtype=tf.float32),
                self.ys_feature: tf.FixedLenSequenceFeature([], dtype=tf.float32),
                self.image_feature: tf.FixedLenSequenceFeature([], dtype=tf.float32),
                self.shape_feature: tf.FixedLenSequenceFeature([], dtype=tf.int64),
            })

        return (context[self.video_id_feature], 
            context[self.image_id_feature], 
            sequence[self.xs_feature], 
            sequence[self.ys_feature], 
            sequence[self.image_feature], 
            sequence[self.shape_feature])


    def prefetch_input_data(self, reader, file_pattern, batch_size,
            values_per_shard, input_queue_capacity_factor=16, num_reader_threads=1,
            shard_queue_name="filename_queue", value_queue_name="input_queue"):
        """Prefetches string values from disk into an input queue.
        In training the capacity of the queue is important because a larger queue
        means better mixing of training examples between shards. The minimum number of
        values kept in the queue is values_per_shard * input_queue_capacity_factor,
        where input_queue_memory factor should be chosen to trade-off better mixing
        with memory usage.
        Args:
            reader: Instance of tf.ReaderBase.
            file_pattern: Comma-separated list of file patterns (e.g.
                /tmp/train_data-?????-of-00100).
            batch_size: Model batch size used to determine queue capacity.
            values_per_shard: Approximate number of values per shard.
            input_queue_capacity_factor: Minimum number of values to keep in the queue
            in multiples of values_per_shard. See comments above.
            num_reader_threads: Number of reader threads to fill the queue.
            shard_queue_name: Name for the shards filename queue.
            value_queue_name: Name for the values input queue.
        Returns:
            A Queue containing prefetched string values.
        """
        data_files = []
        for pattern in file_pattern.split(","):
            data_files.extend(tf.gfile.Glob(pattern))
        if not data_files:
            tf.logging.fatal("Found no input files matching %s", file_pattern)
        else:
            tf.logging.info("Prefetching values from %d files matching %s",
                len(data_files), file_pattern)

        filename_queue = tf.train.string_input_producer(
            data_files, shuffle=True, capacity=16, name=shard_queue_name)
        min_queue_examples = values_per_shard * input_queue_capacity_factor
        capacity = min_queue_examples + 100 * batch_size
        values_queue = tf.RandomShuffleQueue(
            capacity=capacity,
            min_after_dequeue=min_queue_examples,
            dtypes=[tf.string],
            name="random_" + value_queue_name)

        enqueue_ops = []
        for _ in range(num_reader_threads):
            _, value = reader.read(filename_queue)
            enqueue_ops.append(values_queue.enqueue([value]))
        tf.train.queue_runner.add_queue_runner(tf.train.queue_runner.QueueRunner(
            values_queue, enqueue_ops))
        tf.summary.scalar(
            "queue/%s/fraction_of_%d_full" % (values_queue.name, capacity),
            tf.cast(values_queue.size(), tf.float32) * (1. / capacity))

        return values_queue


    def batch_with_dynamic_pad(self, enqueue_list, batch_size, queue_capacity):
        """Batches input images.
        Args:
            images: A list of [video_id, image_id, xs, ys, image, shape]
            batch_size: Batch size.
            queue_capacity: Queue capacity.
        Returns:
            video_ids: tf.int64 Tensor identifying the source videos.
            image_ids: tf.int64 Tensor identifying the source frames.
            xss: tf.float32 Tensor of object points.
            yss: tf.float32 Tensor of object points.
            images: tf.float32 Tensor of flattened image pixels.
            shapes: tf.int64 Tensor of the original image shapes.
        """

        batch = tf.train.batch_join(
            enqueue_list,
            batch_size=batch_size,
            capacity=queue_capacity,
            dynamic_pad=True,
            name="batch_and_pad")

        return batch


    def distort_image(self, image, thread_id):
        """Perform random distortions on an image.
        Args:
            image: A float32 Tensor of shape [height, width, 3] with values in [0, 1).
            thread_id: Preprocessing thread id used to select the ordering of color
            distortions. There should be a multiple of 2 preprocessing threads.
        Returns:
            distorted_image: A float32 Tensor of shape [height, width, 3] with values in
            [0, 1].
        """
        # Randomly flip horizontally.
        with tf.name_scope("flip_horizontal", values=[image]):
            image = tf.image.random_flip_left_right(image)

        # Randomly distort the colors based on thread id.
        color_ordering = thread_id % 2
        with tf.name_scope("distort_color", values=[image]):
            if color_ordering == 0:
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.032)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            elif color_ordering == 1:
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.032)

            # The random_* ops do not necessarily clamp.
            image = tf.clip_by_value(image, 0.0, 1.0)

        return image


    def start(self):
        """Build the batch inputs and distore images for training.
        Returns:
            video_ids: tf.int64 Tensor identifying the source videos.
            image_ids: tf.int64 Tensor identifying the source frames.
            xss: tf.float32 Tensor of object points.
            yss: tf.float32 Tensor of object points.
            images: tf.float32 Tensor of image pixels.
        """

        # Prefetch serialized SequenceExample protos.
        input_queue = self.prefetch_input_data(
            self.reader,
            self.input_file_pattern,
            batch_size=self.batch_size,
            values_per_shard=self.values_per_input_shard,
            input_queue_capacity_factor=self.input_queue_capacity_factor,
            num_reader_threads=self.num_input_reader_threads)

        # Image processing and random distortion. Split across multiple threads
        # with each thread applying a slightly different distortion.
        assert self.num_preprocess_threads % 2 == 0
        enqueue_list = []
        for thread_id in range(self.num_preprocess_threads):
            serialized_sequence_example = input_queue.dequeue()
            video_id, image_id, xs, ys, image, shape = self.parse_sequence_example(
                serialized_sequence_example)
            image = tf.reshape(image, shape)
            image = self.distort_image(image, thread_id=thread_id)
            enqueue_list.append([video_id, image_id, xs, ys, image])

        # Batch inputs.
        queue_capacity = (2 * self.num_preprocess_threads *
                            self.batch_size)
        video_ids, image_ids, xss, yss, images = (
            self.batch_with_dynamic_pad(enqueue_list,
                batch_size=self.batch_size,
                queue_capacity=queue_capacity))

        return video_ids, image_ids, xss, yss, images
