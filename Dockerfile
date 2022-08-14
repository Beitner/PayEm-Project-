# Pull base image
FROM python:3.9
    # Set environment varibles
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
WORKDIR /code/
RUN pip install pipenv 
    # Install dependencies
ADD requirements.txt .
RUN pip install -r requirements.txt
    #Run code
COPY . /code/
CMD ["python", "inference.py"]