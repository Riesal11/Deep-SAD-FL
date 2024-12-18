# how to get clients?
# static vs from server?

# connection to clients

# redistribution of lost client

# TODO: delete data folder in data_distributor

import time
import csv
import binascii
from threading import Timer, Event
from kafka import errors, KafkaProducer, KafkaAdminClient, KafkaConsumer
from kafka.admin import NewTopic

from threading import Thread

def create_topic(topic: NewTopic):
    try:
        admin.create_topics([topic])
    except errors.TopicAlreadyExistsError:
        print("Topic " + topic.name +  " already exists")

def autoclose():
    producer.close()

# use kafka:9092 in container or localhost:29092 on host
admin = KafkaAdminClient(bootstrap_servers='kafka:9092')
print(admin.list_consumer_groups())
server_topic = NewTopic(name='server',
                    num_partitions=1,
                    replication_factor=1)
distributor_topic = NewTopic(name='distributor',
                    num_partitions=1,
                    replication_factor=1)

create_topic(server_topic)
create_topic(distributor_topic)

connected_client_ids = []

# use kafka:9092 in container or localhost:29092 on host
# value_serializer=lambda v: binascii.hexlify(v.encode('utf-8')), 
producer = KafkaProducer(value_serializer=lambda v: binascii.hexlify(v.encode('utf-8')), bootstrap_servers='kafka:9092')

class DistributorPollingThread(Thread):
    def __init__(self, event):
        Thread.__init__(self)
        self.stopped = event
        self.consumer = KafkaConsumer(bootstrap_servers='kafka:9092',
                                value_deserializer=lambda v: binascii.unhexlify(v).decode('utf-8'),
                                # auto_offset_reset='earliest',
                                group_id='my_favorite_group',
                                client_id="distributor",
                                consumer_timeout_ms=1000)
        self.consumer.subscribe([distributor_topic.name])
    
    def run(self):
        while not self.stopped.wait(5.0):
            print("polling thread distributor")
            poll_distributor_topic(self.consumer)

def poll_distributor_topic(consumer: KafkaConsumer):
    batch_size = 500
    records = consumer.poll(10.0)
    print(records)
    for topic_data, consumer_records in records.items():
                for consumer_record in consumer_records:
                    print("Received message: " + consumer_record.value + "\n")
                    print("TODO: spawn data thread")

class DataThread(Thread):
    def __init__(self, event, *args):
        Thread.__init__(self)
        self.stopped = event
        self.client_id = args[0]
        self.topic_name = "topic-client-" + self.client_id
        self.current_index = 0
    
    def run(self):
        while not self.stopped.wait(5.0):
            print("data thread " + self.client_id)
            filename = "data/data"+self.client_id+"_stream.csv"
            with open(filename, "r") as f:
                reader = csv.reader(f, delimiter="\t")
                for i, line in enumerate(reader, self.current_index):
                    delimiter = ","
                    msg = delimiter.join(line)
                    producer.send(self.topic_name, msg)
                self.current_index = i+1

stopFlag_distributor = Event()
distributor_polling_thread = DistributorPollingThread(stopFlag_distributor)
distributor_polling_thread.start()



# t = Timer(15.0, autoclose)
# t.start()

# num_clients = 2

# with open("data/data1_stream.csv", "r") as f:
#     reader = csv.reader(f, delimiter="\t")
#     for i, line in enumerate(reader):
#         delimiter = ","
#         msg = delimiter.join(line)
#         producer.send('my-topic', msg)







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