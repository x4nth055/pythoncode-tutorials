import cv2, argparse, sys


# In this function, we accept an image and convert it to a cartoon form.
def cartoonizer(image_name):
    # Load the image to cartoonize.
    image_to_animate = cv2.imread(image_name)

    # Apply a bilateral filter to smoothen the image while preserving edges.
    smoothened_image = cv2.bilateralFilter(image_to_animate, d=9, sigmaColor=75, sigmaSpace=75)

    # Convert image to gray and create an edge mask using adaptive thresholding.
    gray_image = cv2.cvtColor(smoothened_image, cv2.COLOR_BGR2GRAY)
    edges = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)

    # Combine the smoothened image and the edge mask to create a cartoon-like effect.
    to_cartoon = cv2.bitwise_and(smoothened_image, smoothened_image, mask=edges)

    # Save the cartoon image in our current directory. A new Image would be generated in your current directory.
    cartooned_image = f"cartooned_{image_name}"
    cv2.imwrite(cartooned_image, to_cartoon)

    # Display the result.
    cv2.imshow("Cartooned Image", to_cartoon)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# In this function, we accept user's argument from the terminal. -i or --image to specify the image.
def get_image_argument():
    parser = argparse.ArgumentParser(description="Please specify an image to 'cartoonify'.")
    parser.add_argument('-i', '--image', help="Please use -h or --help to see usage.", dest='image')
    argument = parser.parse_args()

    if not argument.image:
        print("[-] Please specify an image. Use --help to see usage.")
        sys.exit()  # Exit the program

    return argument


# We get the user's input (image) from the terminal and pass it into cartoonizer function.
image_args = get_image_argument()
cartoonizer(image_args.image)
