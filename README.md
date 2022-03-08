# World Pendulum Alliance Tides Post-Processing Script

This is the post-processing Python script that analyses data returned
from the (unattended) `tides.py` pendulum experiment run. It provides 
a web interface in order to show collected data and plot the desired 
graphics. 

## Install procedure

```shell
$ git clone -b parallel https://github.com/bgeneto/wpa-tides.git
$ cd wpa-tides
$ docker build -t tides-streamlit:latest .
```

Check image creation with: 
```shell
$ docker images
```

Run the container with:
```shell
$ docker run -d -p 8501:8501 tides-streamlit:latest
```

Or use the docker compose with the provided `docker-compose.yml` file: 
```shell
$ docker compose up -d
```
