import os
from . import create_app
from .models import Book
from . import db
from flask import jsonify, request, abort

app = create_app(os.getenv('FLASK_CONFIG') or 'default')


@app.route("/book/list", methods=["GET"])
def get_books():
    books = Book.query.all()
    return jsonify([book.to_json() for book in books])


@app.route("/book/<int:isbn>", methods=["GET"])
def get_book(isbn):
    book = Book.query.get(isbn)
    if book is None:
        abort(404)
    return jsonify(book.to_json())


@app.route("/book/<int:isbn>", methods=["DELETE"])
def delete_book(isbn):
    book = Book.query.get(isbn)
    if book is None:
        abort(404)
    db.session.delete(book)
    db.session.commit()
    return jsonify({'result': True})


@app.route('/book', methods=['POST'])
def create_book():
    if not request.json:
        abort(400)
    book = Book(
        title=request.json.get('title'),
        author=request.json.get('author'),
        price=request.json.get('price')
    )
    db.session.add(book)
    db.session.commit()
    return jsonify(book.to_json()), 201


@app.route('/book/<int:isbn>', methods=['PUT'])
def update_book(isbn):
    if not request.json:
        abort(400)
    book = Book.query.get(isbn)
    if book is None:
        abort(404)
    book.title = request.json.get('title', book.title)
    book.author = request.json.get('author', book.author)
    book.price = request.json.get('price', book.price)
    db.session.commit()
    return jsonify(book.to_json())
