# Start from the  base file
FROM opencvcourses/opencv-docker:4.4.0


RUN cd /home/ \
&& apt-get update -qq \
&& apt-get -y install python3-tk

COPY requirements.txt /home/

#Install required libraries
RUN pip install -r requirements.txt


ENV PYTHONPATH="/home/imageprocessingcourse:${PYTHONPATH}"

ENTRYPOINT ["sh", "/home/imageprocessingcourse/evaluator.sh"]


#Clean unnecessary files
RUN rm -rf sampleCode
RUn rm requirements.txt

