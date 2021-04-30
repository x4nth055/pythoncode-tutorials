# URL for our broker used for connecting to the Kafka cluster
KAFKA_BROKER_URL   = "localhost:9092"
# name of the topic hosting the transactions to be processed and requiring processing
TRANSACTIONS_TOPIC = "queuing.transactions"
# these 2 variables will control the amount of transactions automatically generated
TRANSACTIONS_PER_SECOND = float("2.0")
SLEEP_TIME = 1 / TRANSACTIONS_PER_SECOND
# name of the topic hosting the legitimate transactions
LEGIT_TOPIC = "queuing.legit"
# name of the topic hosting the suspicious transactions
FRAUD_TOPIC = "queuing.fraud"