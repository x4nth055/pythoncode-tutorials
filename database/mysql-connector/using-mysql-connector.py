import mysql.connector as mysql
from tabulate import tabulate

# insert MySQL Database information here
HOST = "localhost"
DATABASE = ""
USER = "root"
PASSWORD = ""

# connect to the database
db_connection = mysql.connect(host=HOST, database=DATABASE, user=USER, password=PASSWORD)
# get server information
print(db_connection.get_server_info())
# get the db cursor
cursor = db_connection.cursor()
# get database information
cursor.execute("select database();")
database_name = cursor.fetchone()
print("[+] You are connected to the database:", database_name)
# create a new database called library
cursor.execute("create database if not exists library")

# use that database 
cursor.execute("use library")
print("[+] Changed to `library` database")
# create a table
cursor.execute("""create table if not exists book (
    `id` integer primary key auto_increment not null,
    `name` varchar(255) not null,
    `author` varchar(255) not null,
    `price` float not null,
    `url` varchar(255)
    )""")
print("[+] Table `book` created")

# insert some books
books = [
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

# iterate over books list
for book in books:
    id = book.get("id")
    name = book.get("name")
    author = book.get("author")
    price = book.get("price")
    url = book.get("url")
    # insert each book as a row in MySQL
    cursor.execute("""insert into book (id, name, author, price, url) values (
        %s, %s, %s, %s, %s
    )
    """, params=(id, name, author, price, url))
    print(f"[+] Inserted the book: {name}")

# commit insertion
db_connection.commit()

# fetch the database
cursor.execute("select * from book")
# get all selected rows
rows = cursor.fetchall()
# print all rows in a tabular format
print(tabulate(rows, headers=cursor.column_names))
# close the cursor
cursor.close()
# close the DB connection
db_connection.close()