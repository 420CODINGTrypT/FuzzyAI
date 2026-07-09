#!/bin/bash
# Run MongoDB with authentication enabled and localhost-only binding
# SECURITY: --auth requires users to authenticate. Bind 127.0.0.1 only.

docker run -d \
  -v mongodb_data:/data/db \
  -p 127.0.0.1:27017:27017 \
  --name mongodb \
  mongo:latest \
  mongod --auth --bind_ip 127.0.0.1
