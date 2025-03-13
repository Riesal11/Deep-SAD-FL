# redistribution of lost client

# TODO: delete data folder in data_distributor

import time
import csv
import binascii
import pandas as pd
from threading import Timer, Event
from kafka import errors, KafkaProducer, KafkaAdminClient, KafkaConsumer
from kafka.admin import NewTopic

from threading import Thread

def create_topic(topic: NewTopic):
    try:
        admin.create_topics([topic])
        print("Created topic " + topic.name)
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
backup_ids = []

health_dict = {}

# use kafka:9092 in container or localhost:29092 on host
# value_serializer=lambda v: binascii.hexlify(v.encode('utf-8')), 
producer = KafkaProducer(value_serializer=lambda v: binascii.hexlify(v.encode('utf-8')), 
                         key_serializer=lambda k: binascii.hexlify(k.encode('utf-8')),
                         bootstrap_servers='kafka:9092')

class ServerPollingThread(Thread):
    def __init__(self, event):
        Thread.__init__(self)
        self.stopped = event
        self.consumer = KafkaConsumer(bootstrap_servers='kafka:9092',
                                value_deserializer=lambda v: binascii.unhexlify(v).decode('utf-8'),
                                key_deserializer=lambda k: binascii.unhexlify(k).decode('utf-8'),
                                auto_offset_reset='earliest',
                                client_id="server",
                                consumer_timeout_ms=1000)
        self.consumer.subscribe([server_topic.name])
    
    def run(self):
        while not self.stopped.wait(5.0):
            print("polling thread server")
            poll_server_topic(self.consumer)

def poll_server_topic(consumer: KafkaConsumer):
    records = consumer.poll(10.0)
    for topic_data, consumer_records in records.items():
            for consumer_record in consumer_records:
                print("Received message: " + consumer_record.value + "\n")
                if consumer_record.key == 'new-backup-client':
                    backup_id = str(consumer_record.value)
                    backup_ids.append(backup_id)
                    print("new backup client " + backup_id)
                else:
                    print("unrecognized key")

class DistributorPollingThread(Thread):
    def __init__(self, event):
        Thread.__init__(self)
        self.stopped = event
        self.consumer = KafkaConsumer(bootstrap_servers='kafka:9092',
                                value_deserializer=lambda v: binascii.unhexlify(v).decode('utf-8'),
                                key_deserializer=lambda k: binascii.unhexlify(k).decode('utf-8'),
                                auto_offset_reset='earliest',
                                client_id="distributor",
                                consumer_timeout_ms=1000)
        self.consumer.subscribe([distributor_topic.name])
    
    def run(self):
        while not self.stopped.wait(5.0):
            print("polling thread distributor")
            poll_distributor_topic(self.consumer)

def poll_distributor_topic(consumer: KafkaConsumer):
    records = consumer.poll(10.0)
    for topic_data, consumer_records in records.items():
                for consumer_record in consumer_records:
                    print("Received message: " + consumer_record.value + "\n")
                    if consumer_record.key == 'new-client':
                        client_id = int(consumer_record.value)
                        new_client_topic_name = "client-"+ str(client_id)
                        new_client_topic = NewTopic(name=new_client_topic_name,
                            num_partitions=1,
                            replication_factor=1)
                        create_topic(new_client_topic)
                        stopFlag = Event()
                        data_thread = DataThread(stopFlag, client_id)
                        data_thread.start()
                        thread_dict[client_id] = stopFlag
                    elif consumer_record.key == 'health':
                        client_id = int(consumer_record.value)
                        health_dict[client_id] = time.time()
                    else:
                        print("unrecognized key")

class DataThread(Thread):
    def __init__(self, event, *args):
        Thread.__init__(self)
        self.stopped = event
        self.client_id = str(args[0])
        self.topic_name = "client-" + self.client_id
        self.current_index = 0
        print ("Created data thread for client " + self.client_id)
    
    def run(self):
        while not self.stopped.wait(5.0):
            print("data thread " + self.client_id)
            filename = "data/data"+self.client_id+"_stream.csv"

            # TODO: send slowly, e.g every 30 seconds 500 entries
            # test setup -> different intervals?

            df = pd.read_csv(filename, skiprows=self.current_index, delimiter="\t")
            for index, row in df.iterrows():
                delimiter = ","
                msg = delimiter.join(row)
                producer.send(self.topic_name, key="data", value=msg)
            if len(df.index) > 0:
                self.current_index = len(df.index)
                print ("current index = " + str(self.current_index))

            # with open(filename, "r") as f:
            #     reader = csv.reader(f, delimiter="\t")
            #     for i, line in enumerate(reader, -self.current_index):
            #         delimiter = ","
            #         msg = delimiter.join(line)
            #         producer.send(self.topic_name, key="data", value=msg)
            #     self.current_index = i+1
            #     print ("current index = " + str(self.current_index))

class HealthThread(Thread):
    def __init__(self, event, *args):
        Thread.__init__(self)
        self.stopped = event
        print ("Created health thread")
    
    def run(self):
        while not self.stopped.wait(5.0):
            print ("checking health...")
            now = time.time()
            for key in health_dict:
                print (now-health_dict[key])
                if (now-health_dict[key] > 10):
                    print ("HEALTH CHECK ERROR client id " + str(key))
                    # pick available backup and send message
                    


stopFlag_distributor = Event()
distributor_polling_thread = DistributorPollingThread(stopFlag_distributor)
distributor_polling_thread.start()

stopFlag_distributor = Event()
distributor_polling_thread = ServerPollingThread(stopFlag_distributor)
distributor_polling_thread.start()

stopFlag_health = Event()
health_thread = HealthThread(stopFlag_health)
health_thread.start()

thread_dict = {}







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