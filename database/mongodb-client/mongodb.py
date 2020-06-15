from pymongo import MongoClient
from pprint import pprint

# connect to the MongoDB server
client = MongoClient()
# or explicitly
# client = MongoClient("localhost", 27017)
# list all database names
print("Available databases:", client.list_database_names())
# access the database "python", this will create the actual database
# if it doesn't exist
database = client["python"]
# or this:
# database = client.python
# list all collections
print("Available collections:", database.list_collection_names())
# get books collection (or create one)
books = database["books"]
# insert a single book
result = books.insert_one({
    "name": "Invent Your Own Computer Games with Python, 4E",
    "author": "Al Sweigart",
    "price": 17.99,
    "url": "https://amzn.to/2zxN2W2"
})

print("One book inserted:", result.inserted_id)


# insert many books
books_data = [
    {
        "name": "Automate the Boring Stuff with Python: Practical Programming for Total Beginners",
        "author": "Al Sweigart",
        "price": 17.76,
        "url": "https://amzn.to/2YAncdY"
    },
    {
        "name": "Python Crash Course: A Hands-On, Project-Based Introduction to Programming",
        "author": "Eric Matthes",
        "price": 22.97,
        "url": "https://amzn.to/2yQfQZl"
    },
    {
        "name": "MySQL for Python",
        "author": "Albert Lukaszewski",
        "price": 49.99,
    }
]

result = books.insert_many(books_data)
print("Many books inserted, Ids:", result.inserted_ids)

# get a single book by a specific author
eric_book = books.find_one({"author": "Eric Matthes"})
pprint(eric_book)

# get all books by a specific author
sweigart_books = books.find({"author": "Al Sweigart"})
print("Al Sweigart's books:")
pprint(list(sweigart_books))

# get all documents in books collection
all_books = books.find({})
print("All books:")
pprint(list(all_books))

# delete a specific document by a JSON query
result = books.delete_one({"author": "Albert Lukaszewski"})

# delete all books by Al Sweigart
result = books.delete_many({"author": "Al Sweigart"})

# printing all documents
all_books = books.find({})
print("All books:")
pprint(list(all_books))

# drop this collection
database.drop_collection("books")
# or this:
# books.drop()
# drop this entire database
client.drop_database("python")
# close the connection
client.close()