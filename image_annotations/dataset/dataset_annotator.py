"""Author: Brandon Trabucco
Utility class for human annotating a video dataset.
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import glob.iglob
import imageio
import os.path
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
import time


from image_annotations.video_metadata import VideoMetadata
from image_annotations.abstract import Abstract


class DatasetAnnotator(Abstract):
    """Utility class.
    """

    def __init__(self,
            video_dir,
            num_repetitions):
        """Initialize the class.
        """

        # Search for video filenames
        self.video_files = []
        for filename in glob.iglob(
                os.path.join(video_dir, "*.mp4"), recursive=True):
            self.video_files.append(filename)

        # Load the video files into memory.
        self.num_repetitions = num_repetitions
        self.dimensions = (299, 299)
        self.video_frames = [imageio.get_reader(x,  "ffmpeg") for x in self.video_files]
        print("Finished loading {0} files from {1}".format(
            len(self.video_files), video_dir))

        # Set the coordinate points of interest
        self.xdata, self.ydata = 0.5, 0.5
        self.point_names = ["center"]


    def on_mouse_move(self, event):
        """Listen to the position of the mouse.
        """

        self.xdata, self.ydata = event.xdata, event.ydata


    def start(self):
        """Play the videos in order.
        """

        # Repeat the experiment for data augmentation
        self.final_points = []
        for r in range(self.num_repetitions):

            # Generate placeholder metadata objects
            print("Starting round {0} of {1} metadata.".format(r, self.num_repetitions))
            self.video_points = {video_id: {image_id: VideoMetadata(
                image=scipy.misc.imresize(np.array(img), self.dimensions), 
                image_id=image_id, video_id=video_id, 
                xs=[0.0 for _ in range(len(self.point_names))], 
                ys=[0.0 for _ in range(len(self.point_names))]) 
                for image_id, img in enumerate(v)
                } for video_id, v in enumerate(self.video_frames)}
            print("Finished processing initial videos.")

            # Loop through the interest points and video frames.
            for point_id, point_name in enumerate(self.point_names):
                for video_id, v in enumerate(self.video_frames):
                    fig = plt.figure()
                    fig.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)
                    image_buffer = plt.imshow(scipy.misc.imresize(
                        np.array(v.get_data(0)), self.dimensions))
                    plt.axis("off")
                    fig.show()

                    # Countdown from 3 to 0.
                    for i in reversed(range(3)):
                        print("Starting annotation for {0} in {1}".format(point_name, i))
                        time.sleep(1.0)

                    # Display video frames and collect xs and ys.
                    for image_id, frame in enumerate(v):
                        image_buffer.set_data(scipy.misc.imresize(
                            np.array(frame), self.dimensions))
                        plt.pause(.01)
                        plt.draw()

                        print("video_id={0} image_id={1} x={2} and y={3}".format(
                            video_id, image_id, self.xdata, self.ydata))
                        self.video_points[video_id][image_id].xs[point_id] = self.xdata
                        self.video_points[video_id][image_id].ys[point_id] = self.ydata
                    plt.close()

            # Flatten the image metadata objects to dump in tensorflow.
            print("Collecting round {0} of {1} metadata.".format(r, self.num_repetitions))
            self.final_points.extend([self.video_points[video_id][image_id] 
                for image_id, img in enumerate(v)
                for video_id, v in enumerate(self.video_frames)])
            print("Finished collecting metadata.")
            
        return self.final_points