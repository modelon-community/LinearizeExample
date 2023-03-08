#!/bin/bash
BASEDIR="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
export PYTHONPATH=$PYTHONPATH:$BASEDIR/../Resources/CustomFunctions
cd $BASEDIR
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pytest test*