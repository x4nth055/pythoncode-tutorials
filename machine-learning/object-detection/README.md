# [How to Perform YOLO Object Detection using OpenCV and PyTorch in Python](https://www.thepythoncode.com/article/yolo-object-detection-with-opencv-and-pytorch-in-python)
To run this:
- `pip3 install -r requirements.txt`
- To generate a object detection image on `images/dog.jpg`:
    ```
    python yolov8_opencv.py images/dog.jpg
    ```
    A new image `dog_yolo8.jpg` will appear which has the bounding boxes of different objects in the image.
- For live object detection:
    ```
    python live_yolov8_opencv.py
    ```
- If you want to read from a video file and make predictions:
    ```
    python read_video_yolov8.py 1.mp4
    ```
    This will start detecting objects in that video, in the end, it'll save the resulting video to `output.avi`
- Old files for YOLOv3: `yolo_opencv.py`, `live_yolo_opencv.py`, `read_video.py`
- Feel free to edit the codes for your needs!
