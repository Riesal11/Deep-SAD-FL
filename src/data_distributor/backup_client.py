# connect to server topic
# initiate client.py
# clean topic
# wait until x data
# initiate FL

"""
Testing client Kafka connection and csv writer

prior to running, create data folder in subfolder and add the client initial csv file, in which new data will be appended
"""

import binascii
import uuid
import csv
from kafka import KafkaConsumer, KafkaProducer
import logging
from threading import Thread, Event
import time

import sys
import os


class PollingThread(Thread):
    def __init__(self, event, *args):
        Thread.__init__(self)
        self.stopped = event
        self.consumer: KafkaConsumer = args[0]
        self.backup_id = str(args[1])
        print("backup id: " + self.backup_id)

    def run(self):
        while not self.stopped.wait(5.0):
            self.poll_server()

    def poll_server(self):
        records = self.consumer.poll(10.0)
        for topic_data, consumer_records in records.items():
            for consumer_record in consumer_records:
                print(consumer_record)

# python main.py --fl_mode client --client_id $@ --seed $@ --server_ip_address 10.0.0.20:8080

def main():

    backup_id = str(uuid.uuid4())
    # use kafka:9092 in container or localhost:29092 on host
    # value_serializer=lambda v: binascii.hexlify(v.encode('utf-8')), 
    producer = KafkaProducer(value_serializer=lambda v: binascii.hexlify(v.encode('utf-8')),
                                key_serializer=lambda k: binascii.hexlify(k.encode('utf-8')),
                                bootstrap_servers='10.0.0.20:29092')
    # TODO: generate random identifier
    producer.send('server', key="new-backup-client", value=backup_id)

    consumer = KafkaConsumer(bootstrap_servers='10.0.0.20:29092',
        value_deserializer=lambda v: binascii.unhexlify(v).decode('utf-8'),
        key_deserializer=lambda k: binascii.unhexlify(k).decode('utf-8'),
        client_id=backup_id,
        consumer_timeout_ms=1000)
    consumer.subscribe(["server"])
    stopFlag_poll = Event()
    thread_poll = PollingThread(stopFlag_poll, consumer, backup_id)
    thread_poll.start()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(130)
        except SystemExit:
            os._exit(130)