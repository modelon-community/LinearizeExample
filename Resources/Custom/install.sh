#!/bin/bash
python -m venv venv
source venv/bin/activate
pip install ipykernel
pip install modelon-impact-client==3.0.0.dev11
pip install ipykernel
python -m ipykernel install --user --name=LinearizeVenv