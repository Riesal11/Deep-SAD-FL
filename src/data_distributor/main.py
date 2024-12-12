# how to get clients?
# static vs from server?

# connection to clients

# redistribution of lost client

# TODO: delete data folder in data_distributor

import time
import csv
import binascii
from threading import Timer
from kafka import errors, KafkaProducer, KafkaAdminClient
from kafka.admin import NewTopic


def autoclose():
    producer.close()

# use kafka:9092 in container
admin = KafkaAdminClient(bootstrap_servers='localhost:29092')
print(admin.list_consumer_groups())
server_topic = NewTopic(name='server',
                    num_partitions=1,
                    replication_factor=1)
distributor_topic = NewTopic(name='distributor',
                    num_partitions=1,
                    replication_factor=1)
try:
    admin.create_topics([server_topic, distributor_topic])
except errors.TopicAlreadyExistsError:
    print("Topic already exist")
finally:
    admin.close()

# use kafka:9092 in container
# value_serializer=lambda v: binascii.hexlify(v.encode('utf-8')), 
producer = KafkaProducer(value_serializer=lambda v: binascii.hexlify(v.encode('utf-8')), bootstrap_servers='localhost:29092')

t = Timer(15.0, autoclose)
t.start()

num_clients = 2

with open("data/data1_stream.csv", "r") as f:
    reader = csv.reader(f, delimiter="\t")
    for i, line in enumerate(reader):
        delimiter = ","
        msg = delimiter.join(line)
        producer.send('my-topic', msg)







# # -------------------------------

# #!/usr/bin/env python
# import threading, time

# from kafka import KafkaAdminClient, KafkaConsumer, KafkaProducer
# from kafka.admin import NewTopic


# class Producer(threading.Thread):
#     def __init__(self):
#         threading.Thread.__init__(self)
#         self.stop_event = threading.Event()

#     def stop(self):
#         self.stop_event.set()

#     def run(self):
#         producer = KafkaProducer(bootstrap_servers='localhost:29092')

#         while not self.stop_event.is_set():
#             producer.send('my-topic', b"test")
#             producer.send('my-topic', b"\xc2Hola, mundo!")
#             time.sleep(1)

#         producer.close()


# class Consumer(threading.Thread):
#     def __init__(self):
#         threading.Thread.__init__(self)
#         self.stop_event = threading.Event()

#     def stop(self):
#         self.stop_event.set()

#     def run(self):
#         consumer = KafkaConsumer(bootstrap_servers='localhost:29092',
#                                  auto_offset_reset='earliest',
#                                  consumer_timeout_ms=1000)
#         consumer.subscribe(['my-topic'])

#         while not self.stop_event.is_set():
#             for message in consumer:
#                 print(message)
#                 if self.stop_event.is_set():
#                     break

#         consumer.close()


# def main():
#     # Create 'my-topic' Kafka topic
#     try:
#         admin = KafkaAdminClient(bootstrap_servers='localhost:29092')

#         topic = NewTopic(name='my-topic',
#                          num_partitions=1,
#                          replication_factor=1)
#         admin.create_topics([topic])
#     except Exception:
#         pass

#     tasks = [
#         Producer(),
#         Consumer()
#     ]

#     # Start threads of a publisher/producer and a subscriber/consumer to 'my-topic' Kafka topic
#     for t in tasks:
#         t.start()

#     time.sleep(10)

#     # Stop threads
#     for task in tasks:
#         task.stop()

#     for task in tasks:
#         task.join()


# if __name__ == "__main__":
#     main()