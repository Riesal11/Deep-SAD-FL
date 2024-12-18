#!/bin/bash

echo "Start client $@"
sleep 10
python main.py --fl_mode client --client_id $@ --seed $@ --server_ip_address 10.0.0.20:8080