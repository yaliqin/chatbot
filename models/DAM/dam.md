# How to run
dam_model.py is the interface files to the backend server. It support call from server or test in local

# Docker image build
### First get the base image from chatbot vm
current chatbot vm information is:
gcloud compute --project "core-trees-300107" ssh --zone "us-west2-c" "chatbot"

Project: core-trees-300107  
zone:us-west2-c  
vm name: chatbot

how to get the base image:  
gcloud compute images create dam-base \
        --source-disk chatbot \
        --source-disk-zone us-west2-c \
        --family tf2-latest-gpu

### Build the image with the base image and Dockerfile 

### Run the docker container
sudo docker run --name dam_model -td -p 8080:80 yaliqin/dam:1.0