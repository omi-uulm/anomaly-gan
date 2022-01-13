#!/usr/bin/env sh


# CHANGE FOLDER----------------------------------------------------------------------------------
cd /opt1/program/ || exit


# RUN CODE---------------------------------------------------------------------------------------

ls

api_key=$(python -c "import sys,json; print(json.load(open('../config/wandbkey.json'))['APIKey'])")

export WANDB_API_KEY=$api_key

agent_url=$1

wandb login --relogin $api_key

wandb login

wandb agent $agent_url
