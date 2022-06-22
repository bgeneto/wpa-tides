#!/usr/bin/env python3
import os
import requests
import sys
from dotenv import load_dotenv  

#load_dotenv()
#PORT = os.getenv('PORT')
PORT=8501

# Making a HEAD request
try:
    r = requests.head(f'http://localhost:{PORT}/')
except:
    sys.exit(1)
  
# check status code for response received
# success code - 200
if r.status_code != 200:
    sys.exit(1)

sys.exit(0)
