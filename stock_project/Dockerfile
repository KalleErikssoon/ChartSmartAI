# Author: Karl Eriksson
# base image
FROM python:3.9-slim

# working directory for container
WORKDIR /app

# copy requirements for which libraries are needed for the container
COPY requirements.txt .

# copy pipeline source code into container
COPY . .

# install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# expose default Django port
EXPOSE 8000

# command to run Django server
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]