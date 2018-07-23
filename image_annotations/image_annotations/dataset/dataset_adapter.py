"""Author: Brandon Trabucco
Utility class for serializing ImageMetadata objects into a dataset.
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import numpy as np
import random
import threading
import sys
import os.path
from datetime import datetime


random.seed(1234567)
np.random.seed(1234567)


from image_annotations.abstract import Abstract


class DatasetAdaptor(Abstract):
    """Utility class to serialize the dataset to tensorflow.
    """

    def __init__(self, images, output_dir, num_threads=8):
        """Initiialize the class.
        """

        self.images = images
        self.output_dir = output_dir
        self.num_threads = num_threads


    def start(self):
        """Process the images into tensorflow protos.
        """

        self._process_dataset("train", self.images, 32)


    def _int64_feature(self, value):
        """Wrapper for inserting an int64 Feature into a SequenceExample proto."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

  
    def _float_feature(self, value):
        """Wrapper for inserting a float Feature into a SequenceExample proto."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


    def _bytes_feature(self, value):
        """Wrapper for inserting a bytes Feature into a SequenceExample proto."""
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(value)]))


    def _int64_feature_list(self, values):
        """Wrapper for inserting an int64 FeatureList into a SequenceExample proto."""
        return tf.train.FeatureList(feature=[self._int64_feature(v) for v in values])


    def _float_feature_list(self, values):
        """Wrapper for inserting a float FeatureList into a SequenceExample proto."""
        return tf.train.FeatureList(feature=[self._float_feature(v) for v in values])


    def _bytes_feature_list(self, values):
        """Wrapper for inserting a bytes FeatureList into a SequenceExample proto."""
        return tf.train.FeatureList(feature=[self._bytes_feature(v) for v in values])


    def _to_sequence_example(self, image):
        """Builds a SequenceExample proto for an image.
        Args:
            image: An ImageMetadata object.
        Returns:
            A SequenceExample proto.
        """

        context = tf.train.Features(feature={
            "image/video_id": self._int64_feature(image.video_id),
            "image/image_id": self._int64_feature(image.image_id),
        })
        feature_lists = tf.train.FeatureLists(feature_list={
            "image/xs": self._float_feature_list(image.xs),
            "image/ys": self._float_feature_list(image.ys),
            "image/image":  self._float_feature_list(image.image.flatten()),
            "image/shape":  self._int64_feature_list(image.image.shape)
        })
        sequence_example = tf.train.SequenceExample(
            context=context, feature_lists=feature_lists)
        return sequence_example


    def _process_images(self, thread_index, ranges, name, images, num_shards):
        """Processes and saves a subset of sentences as TFRecord files in one thread.
        Args:
            thread_index: Integer thread identifier within [0, len(ranges)].
            ranges: A list of pairs of integers specifying the ranges of the dataset to
            process in parallel.
            name: Unique identifier specifying the dataset.
            images: List ofImageMetadata.
            num_shards: Integer number of shards for the output files.
        """
        # Each thread produces N shards where N = num_shards / num_threads. For
        # instance, if num_shards = 128, and num_threads = 2, then the first thread
        # would produce shards [0, 64).
        num_threads = len(ranges)
        assert not num_shards % num_threads
        num_shards_per_batch = int(num_shards / num_threads)

        shard_ranges = np.linspace(ranges[thread_index][0], ranges[thread_index][1],
                                    num_shards_per_batch + 1).astype(int)
        num_images_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

        counter = 0
        for s in range(num_shards_per_batch):
            # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
            shard = thread_index * num_shards_per_batch + s
            output_filename = "%s-%.5d-of-%.5d" % (name, shard, num_shards)
            output_file = os.path.join(self.output_dir, output_filename)
            writer = tf.python_io.TFRecordWriter(output_file)

            shard_counter = 0
            images_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
            for i in images_in_shard:
                image = images[i]

                sequence_example = self._to_sequence_example(image)
                if sequence_example is not None:
                    writer.write(sequence_example.SerializeToString())
                    shard_counter += 1
                    counter += 1

                if not counter % 1000:
                    print("%s [thread %d]: Processed %d of %d items in thread batch." %
                        (datetime.now(), thread_index, counter, num_images_in_thread))
                    sys.stdout.flush()

            writer.close()
            print("%s [thread %d]: Wrote %d images to %s" %
                (datetime.now(), thread_index, shard_counter, output_file))
            sys.stdout.flush()
            shard_counter = 0
        print("%s [thread %d]: Wrote %d images to %d shards." %
                (datetime.now(), thread_index, counter, num_shards_per_batch))
        sys.stdout.flush()


    def _process_dataset(self, name, images, num_shards):
        """Processes a complete data set and saves it as a TFRecord.
        Args:
            name: Unique identifier specifying the dataset.
            images: List of ImageMetadata.
            num_shards: Integer number of shards for the output files.
        """
        # Shuffle the ordering of images. Make the randomization repeatable.
        random.seed(12345)
        random.shuffle(images)

        # Break the sentences into num_threads batches. Batch i is defined as
        # sentences[ranges[i][0]:ranges[i][1]].
        num_threads = min(num_shards, self.num_threads)
        spacing = np.linspace(0, len(images), num_threads + 1).astype(np.int)
        ranges = []
        threads = []
        for i in range(len(spacing) - 1):
            ranges.append([spacing[i], spacing[i + 1]])

        # Create a mechanism for monitoring when all threads are finished.
        coord = tf.train.Coordinator()

        # Launch a thread for each batch.
        print("Launching %d threads for spacings: %s" % (num_threads, ranges))
        for thread_index in range(len(ranges)):
            args = (thread_index, ranges, name, images, num_shards)
            t = threading.Thread(target=self._process_images, args=args)
            t.start()
            threads.append(t)

        # Wait for all the threads to terminate.
        coord.join(threads)
        print("%s: Finished processing all %d images in data set '%s'." %
            (datetime.now(), len(images), name))
