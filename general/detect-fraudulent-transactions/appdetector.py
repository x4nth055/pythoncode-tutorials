from flask import Flask, Response, stream_with_context, render_template, json, url_for

from kafka import KafkaConsumer
from settings import *

# create the flask object app
app = Flask(__name__)

def stream_template(template_name, **context):
    print('template name =',template_name)
    app.update_template_context(context)
    t = app.jinja_env.get_template(template_name)
    rv = t.stream(context)
    rv.enable_buffering(5)
    return rv

def is_suspicious(transaction: dict) -> bool:
    """Determine whether a transaction is suspicious."""
    return transaction["amount"] >= 900

# this router will render the template named index.html and will pass the following parameters to it:
# title and Kafka stream
@app.route('/')
def index():
    def g():
        consumer = KafkaConsumer(
            TRANSACTIONS_TOPIC
            , bootstrap_servers=KAFKA_BROKER_URL
            , value_deserializer=lambda value: json.loads(value)
            ,
        )
        for message in consumer:
            transaction: dict = message.value
            topic = FRAUD_TOPIC if is_suspicious(transaction) else LEGIT_TOPIC
            print(topic, transaction)  # DEBUG
            yield topic, transaction

    return Response(stream_template('index.html', title='Fraud Detector / Kafka',data=g()))

if __name__ == "__main__":
   app.run(host="localhost" , debug=True)