import requests

from threading import Thread
from queue import Queue

# thread-safe queue initialization
q = Queue()
# number of threads to spawn
n_threads = 5

# read 1024 bytes every time 
buffer_size = 1024

def download():
    global q
    while True:
        # get the url from the queue
        url = q.get()
        # download the body of response by chunk, not immediately
        response = requests.get(url, stream=True)
        # get the file name
        filename = url.split("/")[-1]
        with open(filename, "wb") as f:
            for data in response.iter_content(buffer_size):
                # write data read to the file
                f.write(data)
        # we're done downloading the file
        q.task_done()


if __name__ == "__main__":
    urls = [
        "https://cdn.pixabay.com/photo/2018/01/14/23/12/nature-3082832__340.jpg",
        "https://cdn.pixabay.com/photo/2013/10/02/23/03/dawn-190055__340.jpg",
        "https://cdn.pixabay.com/photo/2016/10/21/14/50/plouzane-1758197__340.jpg",
        "https://cdn.pixabay.com/photo/2016/11/29/05/45/astronomy-1867616__340.jpg",
        "https://cdn.pixabay.com/photo/2014/07/28/20/39/landscape-404072__340.jpg",
    ] * 5

    # fill the queue with all the urls
    for url in urls:
        q.put(url)

    # start the threads
    for t in range(n_threads):
        worker = Thread(target=download)
        # daemon thread means a thread that will end when the main thread ends
        worker.daemon = True
        worker.start()

    # wait until the queue is empty
    q.join()