import tkinter as tk, numpy as np, cv2, os, face_recognition
from datetime import datetime

# Initialize empty lists to store images and people's names.
known_faces = []
face_labels = []

# Get a list of all images in the TrainingImages directory.
image_files = os.listdir("TrainingImages")

# Loop through the images in the directory.
for image_name in image_files:
    # Read each image and add it to the known_faces list.
    current_image = cv2.imread(f'TrainingImages/{image_name}')
    known_faces.append(current_image)

    # Extract the person's name by removing the file extension and add it to the face_labels list.
    face_labels.append(os.path.splitext(image_name)[0])


# Function to get face encodings from a list of images.
def get_face_encodings(images):
    encoding_list = []
    for image in images:
        # Convert the image to RGB format. RGB is Red Green Blue.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Get the face encoding for the first face found in the image.
        face_encoding = face_recognition.face_encodings(image)[0]
        encoding_list.append(face_encoding)
    return encoding_list


# Define a function to document the recognized face.
def document_recognised_face(name, filename='records.csv'):
    # Get the current date in the YYYY-MM-DD format.
    capture_date = datetime.now().strftime("%Y-%m-%d")

    # Check if the specified CSV file exists.
    if not os.path.isfile(filename):
        # If the file doesn't exist, create it and write the header.
        with open(filename, 'w') as f:
            f.write('Name,Date,Time')  # Create the file and write the header.

    # Open the CSV file for reading and writing ('r+')
    with open(filename, 'r+') as file:
        # Read all lines from the file into a list.
        lines = file.readlines()

        # Extract the names from existing lines in the CSV.
        existing_names = [line.split(",")[0] for line in lines]

        # Check if the provided name is not already in the existing names.
        if name not in existing_names:
            # Get the current time in the HH:MM:SS format.
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")

            # Write the new entry to the CSV file including name, capture date, and time.
            file.write(f'\n{name},{capture_date},{current_time}')


# Get face encodings for known images.
known_face_encodings = get_face_encodings(known_faces)


# Function to start the Facial recognition program.
def start_recognition_program():
    # Open a webcam for capturing video. If you are using your computer's webcam, change 1 to 0.
    # If using an external webcam, leave it as 1.
    video_capture = cv2.VideoCapture(1)

    while True:
        # Read a frame from the webcam.
        frame = video_capture.read()

        # Check if the frame is not None (indicating a successful frame capture).
        if frame is not None:
            frame = frame[1]  # The frame is usually the second element of the tuple returned by video_capture.read().

            # Resize the image to a smaller size.
            resized_frame = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
            resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

            # Detect faces in the current frame.
            face_locations = face_recognition.face_locations(resized_frame)

            # Get face encodings for the faces detected in the current frame.
            current_face_encodings = face_recognition.face_encodings(resized_frame, face_locations)

            # Loop through the detected faces in the current frame.
            for face_encoding, location in zip(current_face_encodings, face_locations):
                # Compare the current face encoding with the known encodings.
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

                # Find the index of the best match. That is, the best resemblance.
                best_match_index = np.argmin(face_distances)

                if matches[best_match_index]:
                    # If a match is found, get the name of the recognized person.
                    recognized_name = face_labels[best_match_index].upper()

                    # Extract face location coordinates.
                    top, right, bottom, left = location
                    top, right, bottom, left = top * 4, right * 4, bottom * 4, left * 4

                    # Draw a rectangle around the recognized face.
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

                    # Draw a filled rectangle and display the name above the face.
                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                    cv2.putText(frame, recognized_name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_COMPLEX, 1,
                                (255, 255, 255), 2)
                    document_recognised_face(recognized_name)

            # Display the image with recognized faces.
            cv2.imshow("Webcam", frame)

        # Check for key press
        key = cv2.waitKey(1) & 0xFF

        # Check if the 'q' key is pressed to exit the program.
        if key == ord('q'):
            break

    # Release the video capture and close all OpenCV windows.
    video_capture.release()
    cv2.destroyAllWindows()


# Create the main application window.
root = tk.Tk()
root.title("Face Recognition Program")

# Create a label
label = tk.Label(root, text="Click the button to start the facial recognition program")
label.pack(pady=10)

# Create a button to start the program
start_button = tk.Button(root, text="Start Recognition", command=start_recognition_program)
start_button.pack(pady=10)


# Function to quit the application. This is for quitting the entire program. To quit the webcam stream, hit q.
def quit_app():
    root.quit()
    cv2.destroyAllWindows()


# Create a quit button to exit the application.
exit_button = tk.Button(root, text="Close", command=quit_app)
exit_button.pack(pady=10)

# Start the Tkinter event loop.
root.mainloop()
