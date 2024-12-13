# # join a consumer group for dynamic partition assignment and offset commits
# from kafka import KafkaConsumer
# consumer = KafkaConsumer(bootstrap_servers='localhost:29092', auto_offset_reset='earliest')
# # or as a static member with a fixed group member name
# # consumer = KafkaConsumer('my_favorite_topic', group_id='my_favorite_group',
# #                          group_instance_id='consumer-1', leave_group_on_close=False)
# print(consumer.bootstrap_connected())
# print(consumer.topics())
# print(consumer.subscription())
# consumer.subscribe(['my-topic'])
# print(consumer.subscription())
# for msg in consumer:
#     print("Got a message")
#     print (msg)

import binascii
import csv
from threading import Timer
from kafka import KafkaConsumer

topic_name = 'my-topic'
# use kafka:9092 in container or localhost:29092 on host
consumer = KafkaConsumer(bootstrap_servers='kafka:9092',
                            value_deserializer=lambda v: binascii.unhexlify(v).decode('utf-8'),
                            # auto_offset_reset='earliest',
                            group_id='my_favorite_group',
                            client_id=1,
                            consumer_timeout_ms=1000)
consumer.subscribe(['my-topic'])
print("Client subscribed to " + topic_name)
def autoclose():
    consumer.close()

t = Timer(30.0, autoclose)
t.start()

with open('fileName.csv', 'w') as f:
    writer = csv.writer(f)
    while True:
        counter = 0
        batch = []
        for message in consumer:
            batch.append(message.value+"\n")
            counter += 1
            if counter >= 1000:
                f.writelines(batch)
                batch = []
                counter = 0
                print("new batch added")

