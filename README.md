# person_tracking
Person tracking demo with an optimized tflite model and opencv tracker


# Setup
Make sure you have Python 3.12+ installed.
Install Poetry: `pip install poetry`
Clone this repository and navigate to the project directory.
Run `poetry install` to install the dependencies.


Alternatively, run `pip install -r requirements.txt`


# Usage
```
python3 person_tracking/track_person.py --video_path <integer for camera id, path to file, or camera url. Default 0>
```


A demo with visualization of person tracking from live camera feed or video file. Press 'q' to exit or stop the script.

The model used currently is a person and body landmark detection model, but only the person bounding box is used. The script tries to keep a track of people based on IOU score (bbox overlap) compared to bounding boxes in previous frame.



Limitations:

The model is capable of detecting upto 6 people. As the script only matches a new inference to previous frame, overlap in people can cause incorrect results.