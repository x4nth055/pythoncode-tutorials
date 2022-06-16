import os
from . import create_app
from .models import Book
from . import db
from flask import jsonify, redirect, request, abort, render_template, url_for

app = create_app(os.getenv('FLASK_CONFIG') or 'default')


@app.route("/")
def index():
    books = Book.query.all()
    return render_template("index.html", books=books)


@app.route("/book/list", methods=["GET"])
def get_books():
    books = Book.query.all()
    return jsonify([book.to_json() for book in books])


@app.route("/book/<int:isbn>", methods=["GET"])
def get_book(isbn):
    book = Book.query.get(isbn)
    if book is None:
        abort(404)
    return render_template("book.html", isbn=isbn)


@app.route("/delete/<int:isbn>", methods=["POST"])
def delete(isbn):
    book = Book.query.get(isbn)
    if book is None:
        abort(404)
    db.session.delete(book)
    db.session.commit()
    return redirect(url_for("index"))


@app.route('/add_book/', methods=['POST'])
def add_book():
    if not request.form:
        abort(400)
    book = Book(
        title=request.form.get('title'),
        author=request.form.get('author'),
        price=request.form.get('price')
    )
    db.session.add(book)
    db.session.commit()
    return redirect(url_for("index"))


@app.route('/update_book/<int:isbn>', methods=['POST'])
def update_book(isbn):
    if not request.form:
        abort(400)
    book = Book.query.get(isbn)
    if book is None:
        abort(404)
    book.title = request.form.get('title', book.title)
    book.author = request.form.get('author', book.author)
    book.price = request.form.get('price', book.price)
    db.session.commit()
    return redirect(url_for("index"))
