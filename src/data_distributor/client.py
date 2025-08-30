"""
Testing client Kafka connection and csv writer

prior to running, create data folder in subfolder and add the client initial csv file, in which new data will be appended
"""

import binascii
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
        self.client_id = str(args[1])
        print("client id: " + self.client_id)

    def run(self):
        while not self.stopped.wait(5.0):
            print("my thread")
            self.poll_distributor()

    def poll_distributor(self):
        batch_size = 500
        records = self.consumer.poll(10.0)
        filename = "../data/wustl_iiot_2021.csv"

        with open(filename, "a") as f:
            writer = csv.writer(f, delimiter=",", lineterminator='\r')
            for topic_data, consumer_records in records.items():
                if topic_data.topic == "client-"+self.client_id:
                    for consumer_record in consumer_records:
                        row = consumer_record.value.split(',')
                        writer.writerow(row)

def main():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    client_id = 1

    # use kafka:9092 in container or localhost:29092 on host
    producer = KafkaProducer(value_serializer=lambda v: binascii.hexlify(v.encode('utf-8')),
                                key_serializer=lambda k: binascii.hexlify(k.encode('utf-8')),
                                bootstrap_servers='<server_ip>:29092')
    producer.send('distributor', key="new-client", value=str(client_id))


    consumer = KafkaConsumer(bootstrap_servers='<server_ip>:29092',
        value_deserializer=lambda v: binascii.unhexlify(v).decode('utf-8'),
        key_deserializer=lambda k: binascii.unhexlify(k).decode('utf-8'),
        client_id=client_id,
        consumer_timeout_ms=1000)
    client_topic = "client-"+ str(client_id)
    while client_topic not in consumer.topics():
        logger.info(consumer.topics())
        time.sleep(1)
    logger.info("new client topic ready to go!")
    consumer.subscribe([client_topic])
    stopFlag = Event()
    thread = PollingThread(stopFlag, consumer, client_id)
    thread.start()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(130)
        except SystemExit:
            os._exit(130)

