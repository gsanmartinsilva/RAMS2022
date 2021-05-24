FROM python:3.8

RUN mkdir /project
WORKDIR /project

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip3 install -U tensorflow-quantum==0.5.0


