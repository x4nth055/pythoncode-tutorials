import sqlite3
import datetime


def create_table():
    db = sqlite3.connect('database.db')
    query = """
    CREATE TABLE if not exists BOOKS
    (ID INTEGER PRIMARY KEY AUTOINCREMENT,
    NAME TEXT NOT NULL,
    CREATED_AT DATETIME default current_timestamp,
    COMPLETED_AT DATATIME 
    )
    """
    cur = db.cursor()
    cur.execute(query)
    db.close()


create_table()


def insert_book(name,  completed_at):
    db = sqlite3.connect('database.db')
    query = """
    INSERT INTO BOOKS(NAME, COMPLETED_AT)

    VALUES (?,?)
    """

    cur = db.cursor()
    cur.execute(query, (name, completed_at))
    db.commit()
    db.close()
    print('completed')


def get_all_books():
    db = sqlite3.connect('database.db')
    statement = 'SELECT id, name, completed_at FROM BOOKS'
    cur = db.cursor()
    items_io = cur.execute(statement)
    item_lst = [i for i in items_io]
    return item_lst


# insert_book('Time, fast or slow', datetime.datetime.now())

def add_book(self):
    title = self.title_input.text()
    if title:
        cursor.execute("INSERT INTO books (title) VALUES (?)", (title,))
        conn.commit()
        self.title_input.clear()
        self.load_books()


def delete_book(book_id):
    # Connect to the SQLite database
    db = sqlite3.connect('database.db')

    # Define the SQL query to delete a book with a specific ID
    query = "DELETE FROM books WHERE id = ?"

    # Execute the query with the provided book ID as a parameter
    db.execute(query, (book_id,))

    # Commit the changes to the database
    db.commit()

    # Close the database connection
    db.close()
