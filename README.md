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