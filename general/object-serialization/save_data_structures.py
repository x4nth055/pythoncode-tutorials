import pickle

# define any Python data structure including lists, sets, tuples, dicts, etc.
l = list(range(10000))

# save it to a file
with open("list.pickle", "wb") as file:
    pickle.dump(l, file)

# load it again
with open("list.pickle", "rb") as file:
    unpickled_l = pickle.load(file)


print("unpickled_l == l: ", unpickled_l == l)
print("unpickled l is l: ", unpickled_l is l)