import cv2, numpy as np, face_recognition, os, tkinter as tk
from tkinter import filedialog

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


# Get face encodings for known images.
known_face_encodings = get_face_encodings(known_faces)


# Function to handle image selection and recognition
def select_and_recognize_image():
    # Use a file dialog to let the user select an image.
    selected_file = filedialog.askopenfilename()
    if selected_file:
        # Read the selected image.
        selected_image = cv2.imread(selected_file)

        # Convert the image to RGB format.
        selected_image_rgb = cv2.cvtColor(selected_image, cv2.COLOR_BGR2RGB)

        # Get face encodings for the selected image.
        selected_face_encodings = face_recognition.face_encodings(selected_image_rgb)

        match_found = False  # Flag to track if a match is found.

        if not selected_face_encodings:
            print("No faces found in the selected image.")
        else:
            # Loop through the detected faces in the selected image.
            for face_encoding in selected_face_encodings:
                # Compare the current face encoding with the known encodings.
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

                # Find the index of the best match. That is, the best resemblance.
                best_match_index = np.argmin(face_distances)

                if matches[best_match_index]:
                    # If a match is found, get the name of the recognized person.
                    recognized_name = face_labels[best_match_index].upper()

                    # Draw a green rectangle around the recognized face.
                    top, right, bottom, left = face_recognition.face_locations(selected_image_rgb)[0]
                    cv2.rectangle(selected_image, (left, top), (right, bottom), (0, 255, 0), 2,)

                    # Display the name below the face.
                    cv2.putText(selected_image, recognized_name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                                (0, 255, 0), 2)

                    match_found = True  # Match found flag.
                    break  # Exit loop as soon as a match is found.

            if not match_found:
                # If no match is found, draw a red rectangle and display No Match.
                top, right, bottom, left = face_recognition.face_locations(selected_image_rgb)[0]
                cv2.rectangle(selected_image, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.putText(selected_image, "No match", (left + 6, bottom - 6), cv2.FONT_HERSHEY_COMPLEX, 1,
                            (0, 0, 255), 2)

            # Show the image with the rectangle and name.
            cv2.imshow("Recognized Image", selected_image)
            known_faces.clear()# To prevent the program from slowing down due to excess unnecessary encodings.
            cv2.waitKey(0)
            cv2.destroyAllWindows()


# Create the main application window.
root = tk.Tk()
root.title("Face Recognition Program")

# Create a button to select an image for recognition.
select_button = tk.Button(root, text="Select Image for Recognition", command=select_and_recognize_image)
select_button.pack(pady=10)


# Function to quit the application.
def quit_app():
    root.quit()


# Create a quit button to exit the application.
quit_button = tk.Button(root, text="Quit", command=quit_app)
quit_button.pack(pady=10)

# Start the Tkinter event loop.
root.mainloop()
