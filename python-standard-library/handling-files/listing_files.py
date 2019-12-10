import os

# print the current directory
print("The current directory:", os.getcwd())

# make an empty directory (folder)
os.mkdir("folder")
# running mkdir again with the same name raises FileExistsError, run this instead:
# if not os.path.isdir("folder"):
#     os.mkdir("folder")
# changing the current directory to 'folder'
os.chdir("folder")
# printing the current directory now
print("The current directory changing the directory to folder:", os.getcwd())

# go back a directory
os.chdir("..")

# make several nested directories
os.makedirs("nested1/nested2/nested3")

# create a new text file
text_file = open("text.txt", "w")
# write to this file some text
text_file.write("This is a text file")

# rename text.txt to renamed-text.txt
os.rename("text.txt", "renamed-text.txt")

# replace (move) this file to another directory
os.replace("renamed-text.txt", "folder/renamed-text.txt")

# print all files and folders in the current directory
print("All folders & files:", os.listdir())

# print all files & folders recursively
for dirpath, dirnames, filenames in os.walk("."):
    # iterate over directories
    for dirname in dirnames:
        print("Directory:", os.path.join(dirpath, dirname))
    # iterate over files
    for filename in filenames:
        print("File:", os.path.join(dirpath, filename))
# delete that file
os.remove("folder/renamed-text.txt")
# remove the folder
os.rmdir("folder")

# remove nested folders
os.removedirs("nested1/nested2/nested3")

open("text.txt", "w").write("This is a text file")

# print some stats about the file
print(os.stat("text.txt"))

# get the file size for example
print("File size:", os.stat("text.txt").st_size)
