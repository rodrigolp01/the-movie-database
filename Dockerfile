FROM python:3
ADD code.py /
ADD requirements.txt /
ADD movies_metadata.csv /
ADD credits.csv /
WORKDIR /
RUN pip install -r requirements.txt
CMD [ "python", "./code.py" ]