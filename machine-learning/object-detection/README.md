# [How to Perform YOLO Object Detection using OpenCV and PyTorch in Python](https://www.thepythoncode.com/article/yolo-object-detection-with-opencv-and-pytorch-in-python)
To run this:
- `pip3 install -r requirements.txt`
- Download the [model weights](https://pjreddie.com/media/files/yolov3.weights) and put them in `weights` folder.
- To generate a object detection image on `images/dog.jpg`:
    ```
    python yolo_opencv.py images/dog.jpg
    ```
    A new image `dog_yolo3.jpg` will appear which has the bounding boxes of different objects in the image.
- For live object detection:
    ```
    python live_yolo_opencv.py
    ```
- If you want to read from a video file and make predictions:
    ```
    python read_video.py video.avi
    ```
    This will start detecting objects in that video, in the end, it'll save the resulting video to `output.avi`
- If you wish to use PyTorch for GPU acceleration, please install PyTorch CUDA [here](https://pytorch.org/get-started) and use `yolo.py` file.
- Feel free to edit the codes for your needs!
