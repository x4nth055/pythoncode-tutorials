from flask import Flask
from flask_restful import Api
from models import db
import config
from resources import TaskList

# Create the Flask application and the Flask-RESTful API manager.
app = Flask(__name__)
app.config.from_object(config)
# Initialize the Flask-SQLAlchemy object.
db.init_app(app)
# Create the Flask-RESTful API manager.
api = Api(app)
# Create the endpoints.
api.add_resource(TaskList, '/tasks')

if __name__ == '__main__':
    # Create the database tables.
    with app.app_context():
        db.create_all()
    # Start the Flask development web server.
    app.run(debug=True)
