import tarfile
from tqdm import tqdm # pip3 install tqdm


def decompress(tar_file, path, members=None):
    """
    Extracts `tar_file` and puts the `members` to `path`.
    If members is None, all members on `tar_file` will be extracted.
    """
    tar = tarfile.open(tar_file, mode="r:gz")
    if members is None:
        members = tar.getmembers()
    # with progress bar
    # set the progress bar
    progress = tqdm(members)
    for member in progress:
        tar.extract(member, path=path)
        # set the progress description of the progress bar
        progress.set_description(f"Extracting {member.name}")
    # or use this
    # tar.extractall(members=members, path=path)
    # close the file
    tar.close()


def compress(tar_file, members):
    """
    Adds files (`members`) to a tar_file and compress it
    """
    # open file for gzip compressed writing
    tar = tarfile.open(tar_file, mode="w:gz")
    # with progress bar
    # set the progress bar
    progress = tqdm(members)
    for member in progress:
        # add file/folder/link to the tar file (compress)
        tar.add(member)
        # set the progress description of the progress bar
        progress.set_description(f"Compressing {member}")
    # close the file
    tar.close()


# compress("compressed.tar.gz", ["test.txt", "test_folder"])
# decompress("compressed.tar.gz", "extracted")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="TAR file compression/decompression using GZIP.")
    parser.add_argument("method", help="What to do, either 'compress' or 'decompress'")
    parser.add_argument("-t", "--tarfile", help="TAR file to compress/decompress, if it isn't specified for compression, the new TAR file will be named after the first file to compress.")
    parser.add_argument("-p", "--path", help="The folder to compress into, this is only for decompression. Default is '.' (the current directory)", default="")
    parser.add_argument("-f", "--files", help="File(s),Folder(s),Link(s) to compress/decompress separated by ','.")

    args = parser.parse_args()
    method = args.method
    tar_file = args.tarfile
    path = args.path
    files = args.files

    # split by ',' to convert into a list
    files = files.split(",") if isinstance(files, str) else None

    if method.lower() == "compress":
        if not files:
            print("Files to compress not provided, exiting...")
            exit(1)
        elif not tar_file:
            # take the name of the first file
            tar_file = f"{files[0]}.tar.gz"
        compress(tar_file, files)
    elif method.lower() == "decompress":
        if not tar_file:
            print("TAR file to decompress is not provided, nothing to do, exiting...")
            exit(2)
        decompress(tar_file, path, files)
    else:
        print("Method not known, please use 'compress/decompress'.")

