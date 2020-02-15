import requests
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter

# number of threads to spawn
n_threads = 5

# read 1024 bytes every time 
buffer_size = 1024


def download(url):
    # download the body of response by chunk, not immediately
    response = requests.get(url, stream=True)
    # get the file name
    filename = url.split("/")[-1]
    with open(filename, "wb") as f:
        for data in response.iter_content(buffer_size):
            # write data read to the file
            f.write(data)


if __name__ == "__main__":
    urls = [
        "https://cdn.pixabay.com/photo/2018/01/14/23/12/nature-3082832__340.jpg",
        "https://cdn.pixabay.com/photo/2013/10/02/23/03/dawn-190055__340.jpg",
        "https://cdn.pixabay.com/photo/2016/10/21/14/50/plouzane-1758197__340.jpg",
        "https://cdn.pixabay.com/photo/2016/11/29/05/45/astronomy-1867616__340.jpg",
        "https://cdn.pixabay.com/photo/2014/07/28/20/39/landscape-404072__340.jpg",
    ] * 5

    t = perf_counter()
    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        pool.map(download, urls)
    print(f"Time took: {perf_counter() - t:.2f}s")