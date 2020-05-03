# [How to Blur Faces in Images using OpenCV in Python](https://www.thepythoncode.com/article/blur-faces-in-images-using-opencv-in-python)
To run this:
- `pip3 install -r requirements.txt`
- To blur faces of the image `father-and-daughter.jpg`:
    ```
    python blur_faces.py father-and-daughter.jpg
    ```
    This should show the blurred image and save it of the name `image_blurred.jpg` in your current directory.

- To blur faces using your live camera:
    ```
    python blur_faces_live.py
    ```
- To blur faces of a video:
    ```
    python blur_faces_video.py video.3gp
    ```