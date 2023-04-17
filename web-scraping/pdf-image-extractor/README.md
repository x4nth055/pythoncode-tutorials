# [How to Extract Images from PDF in Python](https://www.thepythoncode.com/article/extract-pdf-images-in-python)
To run this:
- `pip3 install -r requirements.txt`
- To extract and save all images of `1710.05006.pdf` PDF file, you run:
    ```
    python pdf_image_extractor.py 1710.05006.pdf
    ```
    This will save all available images in the current directory and outputs:
    ```
    [!] No images found on page 0
    [+] Found a total of 3 images in page 1
    [+] Found a total of 3 images in page 2
    [!] No images found on page 3
    [!] No images found on page 4
    ```
- To extract and save all images of 800x800 and higher of `1710.05006.pdf` PDF file, and save them in `images` directory in the PNG format, you run:
    ```
    python pdf_image_extractor_cli.py 1710.05006.pdf -o extracted-images -f png -w 800 -he 800
    ```
    This will save all available images in the `images` directory and outputs:
    ```
    [!] No images found on page 0
    [+] Found a total of 3 images in page 1
    [-] Skipping image 1 on page 1 due to its small size.
    [-] Skipping image 2 on page 1 due to its small size.
    [-] Skipping image 3 on page 1 due to its small size.
    [+] Found a total of 3 images in page 2
    [-] Skipping image 2 on page 2 due to its small size.
    [!] No images found on page 3
    [!] No images found on page 4
    ```