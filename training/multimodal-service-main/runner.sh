#!/bin/bash
# Kill process before
kill -TERM $(ps aux | grep 'middleware' | awk '{print $2}')
kill -TERM $(ps aux | grep 'webui.py' | awk '{print $2}')

# SET ENV variables
export BEE_PORT=52782
export MIDDLEWARE_ENV=local
export GR_PORT=27777

# Change directory to middleware
cd middleware
export GO111MODULE=on
bash run.sh &

# Change directory to gradio_web
cd ../gradio_web
bash run.sh
