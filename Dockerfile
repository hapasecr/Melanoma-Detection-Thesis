FROM python:3.7-alpine

RUN adduser -D melanomadetector

WORKDIR /home/melanomadetector

RUN apk --no-cache --update-cache add gcc gfortran python python-dev py-pip build-base wget freetype-dev libpng-dev openblas-dev
RUN ln -s /usr/include/locale.h /usr/include/xlocale.h

COPY requirements.txt requirements.txt
RUN python -m venv venv
RUN venv/bin/pip install --upgrade pip setuptools 
RUN venv/bin/pip3 install --no-cache-dir -r requirements.txt
# RUN venv/bin/pip install numpy==1.14.3

COPY app app
COPY appMain.py config.py ./

COPY featext featext
COPY mlmodels mlmodels
COPY preprocessing preprocessing
COPY util util

COPY ip ip
COPY results/op results/op

COPY dataset.npz testcase.npz ./

COPY boot.sh ./
RUN chmod +x boot.sh

ENV FLASK_APP appMain.py

RUN chown -R melanomadetector:melanomadetector ./
USER melanomadetector

EXPOSE 5000
ENTRYPOINT ["./boot.sh"]
