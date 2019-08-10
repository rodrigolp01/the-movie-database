FROM python:3
ADD code.py /
ADD content_recommender.py /
ADD similarity_content_recommender.py /
ADD requirements.txt /
ADD movies_metadata.csv /
ADD credits.csv /
WORKDIR /
RUN pip install -r requirements.txt
CMD [ "python", "./code.py" ]