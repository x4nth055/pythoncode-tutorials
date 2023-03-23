from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Task(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    description = db.Column(db.String(200), nullable=False) # nullable=False means that the column cannot be empty

    def __repr__(self):
        # This method is used to print the object.
        return f'Task {self.id}: {self.description}'
