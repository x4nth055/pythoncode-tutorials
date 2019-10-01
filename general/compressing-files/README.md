# [How to Compress and Decompress Files in Python](https://www.thepythoncode.com/article/compress-decompress-files-tarfile-python)
To run this:
- `pip3 install -r requirements.txt`
- 
    ```
    python tar.py --help
    ```
    **Output:**
    ```
    usage: tar.py [-h] [-t TARFILE] [-p PATH] [-f FILES] method

    TAR file compression/decompression using GZIP.

    positional arguments:
    method                What to do, either 'compress' or 'decompress'

    optional arguments:
    -h, --help            show this help message and exit
    -t TARFILE, --tarfile TARFILE
                            TAR file to compress/decompress, if it isn't specified
                            for compression, the new TAR file will be named after
                            the first file to compress.
    -p PATH, --path PATH  The folder to compress into, this is only for
                            decompression. Default is '.' (the current directory)
    -f FILES, --files FILES
                            File(s),Folder(s),Link(s) to compress/decompress
                            separated by ','.
    ```
- If you want to compress one or more file(s)/folder(s):
    ```
    python tar.py compress -f test_folder,test.txt
    ```
    This will compress the folder `test_folder` and the file `test.txt` into a single TAR compressed file named: `test_folder.tar.gz`
    If you want to name the TAR file yourself, consider using `-t` flag.
- If you want to decompress a TAR file named `test_folder.tar.gz` into a new folder called `extracted` for instance:
    ```
    python tar.py decompress -t test_folder.tar.gz -p extracted
    ```
    A new folder `extracted` will appear that contains everything on `test_folder.tar.gz` decompressed.