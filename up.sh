#!/usr/bin/env bash

docker stop peepholebot
docker rm peepholebot
docker build -t peepholebot .
docker run -d --restart unless-stopped -v $(pwd):/app --env-file .env peepholebot