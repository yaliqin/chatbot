FROM tensorflow/tensorflow:2.4.1-gpu

MAINTAINER Yali Qin "qinyali@gmail.com"

RUN apt-get update -y && \
    apt-get install -y python3-pip python3-dev

RUN mkdir -p /home/ally/github/chatbot

WORKDIR /home/ally/github/chatbot

COPY ./requirements.txt ./requirements.txt

#WORKDIR /

RUN pip3 install -r ./requirements.txt

RUN mkdir -p models/DAM
COPY models/DAM/* ./models/DAM/

RUN mkdir data
COPY data/* ./data/

RUN mkdir -p preprocess
COPY preprocess/* ./preprocess/


#ENTRYPOINT [ "python3" ]
RUN cd models/DAM
CMD ["sh","-c","run.sh" ]
