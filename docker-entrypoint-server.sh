#!/bin/bash

echo "Start server"
mkdir -p ../data
python main.py --fl_mode server --server_ip_address [::]:8080