"""Author: Brandon Trabuco
Object to house the identifiers for an image to be saved.
"""


from collections import namedtuple


VideoMetadata = namedtuple("VideoMetadata", ["image", "image_id", 
    "video_id", "xs", "ys"])
