# [Age Prediction using OpenCV in Python](https://www.thepythoncode.com/article/predict-age-using-opencv)
Before running the code, do the following:
- `pip3 install -r requirements.txt`
- Download [face detection](https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel) and [age detection](https://drive.google.com/open?id=1kiusFljZc9QfcIYdU2s7xrtWHTraHwmW) models and put them in the `weights` folder.
- Run the program:
    ```
    python predict_age.py 3-people.jpg
    ```