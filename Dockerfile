FROM python:3.6-alpine

RUN adduser -D melanomadetector

WORKDIR /home/melanomadetector

COPY requirements.txt requirements.txt
RUN python -m venv venv
RUN venv/bin/pip install -r requirements.txt

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
