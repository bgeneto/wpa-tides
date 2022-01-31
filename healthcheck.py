#!/usr/bin/env python3
import requests
import sys
  
# Making a HEAD request
try:
    r = requests.head('http://localhost:8501/')
except:
    sys.exit(1)
  
# check status code for response received
# success code - 200
if r.status_code != 200:
    sys.exit(1)

sys.exit(0)
