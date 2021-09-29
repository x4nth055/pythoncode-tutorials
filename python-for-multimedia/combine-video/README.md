# [How to Concatenate Video Files in Python](https://www.thepythoncode.com/article/concatenate-video-files-in-python)
To run this:
- `pip3 install -r requirements.txt`
- 
```
    $ python concatenate_video.py --help
```
**Output**:
```
    usage: concatenate_video.py [-h] [-c CLIPS [CLIPS ...]] [-r REDUCE] [-o OUTPUT]

    Simple Video Concatenation script in Python with MoviePy Library

    optional arguments:
    -h, --help            show this help message and exit
    -c CLIPS [CLIPS ...], --clips CLIPS [CLIPS ...]
                            List of audio or video clip paths
    -r REDUCE, --reduce REDUCE
                            Whether to use the `reduce` method to reduce to the lowest quality on the resulting clip
    -o OUTPUT, --output OUTPUT
                            Output file name
```
- To combine `zoo.mp4` and `directed-by-robert.mp4` to produce `output.mp4`:
```
    $ python concatenate_video.py -c zoo.mp4 directed-by-robert.mp4 -o output.mp4
```