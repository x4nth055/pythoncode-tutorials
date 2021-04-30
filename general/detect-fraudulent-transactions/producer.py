import os
import json
from time import sleep
from kafka import KafkaProducer
# import initialization parameters
from settings import *
from transactions import create_random_transaction


if __name__ == "__main__":
   producer = KafkaProducer(bootstrap_servers = KAFKA_BROKER_URL
                            #Encode all values as JSON
                           ,value_serializer = lambda value: json.dumps(value).encode()
                           ,)
   while True:
       transaction: dict = create_random_transaction()
       producer.send(TRANSACTIONS_TOPIC, value= transaction)
       print(transaction) #DEBUG
       sleep(SLEEP_TIME)