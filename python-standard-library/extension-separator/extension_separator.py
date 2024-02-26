import os
import glob
import shutil

# dictionary mapping each extension with its corresponding folder
# For example, 'jpg', 'png', 'ico', 'gif', 'svg' files will be moved to 'images' folder
# feel free to change based on your needs
extensions = {
    "jpg": "images",
    "png": "images",
    "ico": "images",
    "gif": "images",
    "svg": "images",
    "jfif": "images",
    "sql": "sql",
    "exe": "programs",
    "msi": "programs",
    "pdf": "pdf",
    "epub": "epub",
    "xlsx": "excel",
    "csv": "excel",
    "rar": "archive",
    "zip": "archive",
    "gz": "archive",
    "tar": "archive",
    "7z": "archive",
    "docx": "word",
    "torrent": "torrent",
    "txt": "text",
    "log": "text",
    "md": "text",
    "ipynb": "python",
    "py": "python",
    "pptx": "powerpoint",
    "ppt": "powerpoint",
    "mp3": "audio",
    "wav": "audio",
    "mp4": "video",
    "m3u8": "video",
    "webm": "video",
    "ts": "video",
    "avi": "video",
    "json": "json",
    "css": "web",
    "js": "web",
    "html": "web",
    "webp": "web",
    "apk": "apk",
    "sqlite3": "sqlite3",
}


if __name__ == "__main__":
    path = r"E:\Downloads"
    # setting verbose to 1 (or True) will show all file moves
    # setting verbose to 0 (or False) will show basic necessary info
    verbose = 0
    for extension, folder_name in extensions.items():
        # get all the files matching the extension
        files = glob.glob(os.path.join(path, f"*.{extension}"))
        print(f"[*] Found {len(files)} files with {extension} extension")
        if not os.path.isdir(os.path.join(path, folder_name)) and files:
            # create the folder if it does not exist before
            print(f"[+] Making {folder_name} folder")
            os.mkdir(os.path.join(path, folder_name))
        for file in files:
            # for each file in that extension, move it to the correponding folder
            basename = os.path.basename(file)
            dst = os.path.join(path, folder_name, basename)
            if verbose:
                print(f"[*] Moving {file} to {dst}")
            try:
                shutil.move(file, dst)
            except Exception as e:
                print(f"[!] Error: {e}")
                continue