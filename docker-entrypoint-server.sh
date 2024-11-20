#!/bin/bash

echo "Start server"
cd ..
mkdir data
cd src
python main.py --fl_mode server --server_ip_address [::]:8080