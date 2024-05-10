#!/bin/bash


echo "CURRENT PATH             = $(pwd)"

HOSTPATH=$(pwd)
CONTAINERPATH=/home/imageprocessingcourse


echo "HOST BINDMOUNT PATH         = ${HOSTPATH}"
echo "CONTAINER BINDMOUNT PATH    = ${CONTAINERPATH}"


docker build -t cse2225image:latest .


docker rm cse2225container
docker run \
--name cse2225container \
-it \
-p 1984:1984 \
-v $HOSTPATH:$CONTAINERPATH:rw \
-v $XSOCK:$XSOCK:rw \
-v $XAUTH:$XAUTH:rw \
-e DISPLAY=host.docker.internal:0 \
-e XAUTHORITY=$XAUTH \
--ipc="host" \
cse2225image:latest 

