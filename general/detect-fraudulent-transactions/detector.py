import os
import json
from kafka import KafkaConsumer, KafkaProducer
from settings import *

def is_suspicious(transaction: dict) -> bool:
    """Simple condition to determine whether a transaction is suspicious."""
    return transaction["amount"] >= 900

if __name__ == "__main__":
   consumer = KafkaConsumer(
       TRANSACTIONS_TOPIC
      ,bootstrap_servers=KAFKA_BROKER_URL
      ,value_deserializer = lambda value: json.loads(value)
      ,
   )

   for message in consumer:
       transaction: dict = message.value
       topic = FRAUD_TOPIC if is_suspicious(transaction) else LEGIT_TOPIC
       print(topic,transaction) #DEBUG