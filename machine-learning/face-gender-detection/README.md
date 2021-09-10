# [Gender Detection using OpenCV in Python](https://www.thepythoncode.com/article/gender-detection-using-opencv-in-python)
Before running the code, do the following:
- `pip3 install -r requirements.txt`
- Download [face detection](https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel) and [gender detection](https://drive.google.com/open?id=1W_moLzMlGiELyPxWiYQJ9KFaXroQ_NFQ) models and put them in the `weights` folder. Check [the tutorial](https://www.thepythoncode.com/article/gender-detection-using-opencv-in-python) for more on how to set it up.
- Run the program:
    ```
    python predict_gender.py "images\\Donald Trump.jpg"
    ```