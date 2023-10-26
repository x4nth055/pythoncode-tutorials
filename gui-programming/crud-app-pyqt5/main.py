from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QScrollArea,
                             QLineEdit, QFormLayout, QHBoxLayout, QFrame, QDateEdit,
                             QPushButton, QLabel, QListWidget, QDialog, QAction, QToolBar)
from PyQt5.QtCore import Qt

from datetime import datetime
from db import (get_all_books, create_table, insert_book, delete_book)


class CreateRecord(QFrame):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window  # Pass a reference to the main window

        self.date_entry = QDateEdit()
        self.book_name = QLineEdit()
        self.book_name.setPlaceholderText('Book name')
        self.add_button = QPushButton(text="Add Book")
        # Connect the button to add_book function
        self.add_button.clicked.connect(self.add_book)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel('Book Name:'))
        layout.addWidget(self.book_name)
        layout.addWidget(QLabel('Completed Date:'))
        layout.addWidget(self.date_entry)
        layout.addWidget(self.add_button)

    def add_book(self):
        book_name = self.book_name.text()
        completed_date = self.date_entry.date().toString("yyyy-MM-dd")

        if book_name:
            insert_book(book_name, completed_date)
            # Reload the book collection after adding a book
            self.main_window.load_collection()
            self.book_name.clear()  # Clear the input field


class BookCard(QFrame):
    def __init__(self, book_id, bookname, completed_date):
        super().__init__()
        self.setStyleSheet(
            'background:white; border-radius:4px; color:black;'
        )
        self.setFixedHeight(110)
        self.book_id = book_id
        layout = QVBoxLayout()
        label = QLabel(f'<strong>{bookname}</strong>')

        # Update the format string here
        parsed_datetime = datetime.strptime(completed_date, "%Y-%m-%d")
        formatted_datetime = parsed_datetime.strftime("%Y-%m-%d")

        date_completed = QLabel(f"Completed {formatted_datetime}")
        delete_button = QPushButton(
            text='Delete', clicked=self.delete_book_click)
        # delete_button.setFixedWidth(60)
        delete_button.setStyleSheet('background:red; padding:4px;')

        layout.addWidget(label)
        layout.addWidget(date_completed)
        layout.addWidget(delete_button)
        layout.addStretch()
        self.setLayout(layout)

    def delete_book_click(self):
        delete_book(self.book_id)
        self.close()


class Main(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.load_collection()

    def initUI(self):
        self.main_frame = QFrame()
        self.main_layout = QVBoxLayout(self.main_frame)

        # add register widget
        # Pass a reference to the main window
        self.register_widget = CreateRecord(self)
        self.main_layout.addWidget(self.register_widget)

        books_label = QLabel('Completed Books')
        books_label.setStyleSheet('font-size:18px;')
        self.main_layout.addWidget(books_label)
        self.book_collection_area()

        self.setCentralWidget(self.main_frame)

    def book_collection_area(self):
        scroll_frame = QFrame()
        self.book_collection_layout = QVBoxLayout(scroll_frame)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(scroll_frame)
        scroll.setStyleSheet('QScrollArea{border:0px}')

        self.book_collection_layout.addStretch()
        self.main_layout.addWidget(scroll)

    def load_collection(self):
        # Clear existing book cards before reloading
        for i in reversed(range(self.book_collection_layout.count())):
            widget = self.book_collection_layout.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()

        collections = get_all_books()
        for collection in collections:
            frame = BookCard(*collection)
            self.book_collection_layout.insertWidget(0, frame)


def main():
    app = QApplication([])
    app.setStyle('fusion')
    win = Main()
    win.show()
    app.exec_()


if __name__ == '__main__':
    main()
