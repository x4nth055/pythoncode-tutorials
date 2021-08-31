# Import Libraries
import cv2
import os
import filetype
import numpy as np

# The model architecture
# download from: https://drive.google.com/open?id=1kiusFljZc9QfcIYdU2s7xrtWHTraHwmW
AGE_MODEL = 'weights/deploy_age.prototxt'
# The model pre-trained weights
# download from: https://drive.google.com/open?id=1kWv0AjxGSN0g31OeJa02eBGM0R_jcjIl
AGE_PROTO = 'weights/age_net.caffemodel'
# Each Caffe Model impose the shape of the input image also image preprocessing is required like mean
# substraction to eliminate the effect of illunination changes
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
# Represent the 8 age classes of this CNN probability layer
AGE_INTERVALS = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)',
                 '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
# download from: https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt
FACE_PROTO = "weights/deploy.prototxt.txt"
# download from: https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel
FACE_MODEL = "weights/res10_300x300_ssd_iter_140000_fp16.caffemodel"

# Initialize frame size
frame_width = 1280
frame_height = 720

# load face Caffe model
face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)
# Load age prediction model
age_net = cv2.dnn.readNetFromCaffe(AGE_MODEL, AGE_PROTO)


def get_faces(frame, confidence_threshold=0.5):
    """Returns the box coordinates of all detected faces"""
    # convert the frame into a blob to be ready for NN input
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 177.0, 123.0))
    # set the image as input to the NN
    face_net.setInput(blob)
    # perform inference and get predictions
    output = np.squeeze(face_net.forward())
    # initialize the result list
    faces = []
    # Loop over the faces detected
    for i in range(output.shape[0]):
        confidence = output[i, 2]
        if confidence > confidence_threshold:
            box = output[i, 3:7] * np.array([frame_width, frame_height, frame_width, frame_height])
            # convert to integers
            start_x, start_y, end_x, end_y = box.astype(np.int)
            # widen the box a little
            start_x, start_y, end_x, end_y = start_x - \
                10, start_y - 10, end_x + 10, end_y + 10
            start_x = 0 if start_x < 0 else start_x
            start_y = 0 if start_y < 0 else start_y
            end_x = 0 if end_x < 0 else end_x
            end_y = 0 if end_y < 0 else end_y
            # append to our list
            faces.append((start_x, start_y, end_x, end_y))
    return faces


def display_img(title, img):
    """Displays an image on screen and maintains the output until the user presses a key"""
    # Display Image on screen
    cv2.imshow(title, img)
    # Mantain output until user presses a key
    cv2.waitKey(0)
    # Destroy windows when user presses a key
    cv2.destroyAllWindows()


def predict_age(input_path: str):
    """Predict the age of the faces showing in the image"""
    # Read Input Image
    img = cv2.imread(input_path)
    # resize the image
    img = cv2.resize(img, (frame_width, frame_height))
    # Take a copy of the initial image and resize it
    frame = img.copy()
    faces = get_faces(frame)
    for i, (start_x, start_y, end_x, end_y) in enumerate(faces):
            face_img = frame[start_y: end_y, start_x: end_x]
            # image --> Input image to preprocess before passing it through our dnn for classification.
            blob = cv2.dnn.blobFromImage(
                image=face_img, scalefactor=1.0, size=(227, 227), 
                mean=MODEL_MEAN_VALUES, swapRB=False
            )
            # Predict Age
            age_net.setInput(blob)
            age_preds = age_net.forward()
            print("="*30, f"Face {i+1} Prediction Probabilities", "="*30)
            for i in range(age_preds[0].shape[0]):
                print(f"{AGE_INTERVALS[i]}: {age_preds[0, i]*100:.2f}%")
            i = age_preds[0].argmax()
            age = AGE_INTERVALS[i]
            age_confidence_score = age_preds[0][i]
            # Draw the box
            label = f"Age:{age} - {age_confidence_score*100:.2f}%"
            print(label)
            # get the position where to put the text
            yPos = start_y - 15
            while yPos < 15:
                yPos += 15
            # write the text into the frame
            cv2.putText(frame, label, (start_x, yPos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), thickness=2)
            # draw the rectangle around the face
            cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), color=(255, 0, 0), thickness=2)
    # Display processed image
    display_img('Age Estimator', frame)
    # save the image if you want
    # cv2.imwrite("predicted_age.jpg", frame)



if __name__ == '__main__':
    # Parsing command line arguments entered by user
    import sys
    image_path = sys.argv[1]
    predict_age(image_path)
