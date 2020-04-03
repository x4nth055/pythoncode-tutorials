from github import Github

# your github account credentials
username = "username"
password = "password"
# initialize github object
g = Github(username, password)

# searching for my repository
repo = g.search_repositories("pythoncode tutorials")[0]

# create a file and commit n push
repo.create_file("test.txt", "commit message", "content of the file")

# delete that created file
contents = repo.get_contents("test.txt")
repo.delete_file(contents.path, "remove test.txt", contents.sha)