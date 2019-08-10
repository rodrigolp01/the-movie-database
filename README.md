# The Movie Dataset Challenge

Code used to forecasting movies revenue and classification movie success of the dataset (https://www.kaggle.com/rounakbanik/the-movies-dataset)

# Instalation

For python 3.6.8:

```bash
pip install -r requirements.txt
```

## Usage locally

From terminal, run the command to generate predictions, classifications, images related to features analysis:

```bash
python code.py
```
Run this other commands to recommend by movie genre and by movie title, respectively:

```bash
python content_recommender.py
```

```bash
python similarity_content_recommender.py
```

## Usage with DockerFile

From terminal, inside a folder with code.py, requirements.txt, Dockerfile, Csv files:

```bash
sudo docker build -t the-movie .
```

This command will create the image. Now you can run this image with:

```bash
sudo docker run the-movie
```

And copy a result image(to visualize in the host) with this command:

```bash
sudo docker cp <container-id>:/<image> .
```
The above command will only work if the container is active.

To call the other scripts you can activate the container and access it

```bash
sudo docker start <container-id>
```

```bash
sudo docker run -it the-movie /bin/bash
```
