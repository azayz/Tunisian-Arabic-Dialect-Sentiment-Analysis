FROM python:3.8.3

COPY requirements.txt .

RUN pip3 install --upgrade pip

RUN pip3 install -r requirements.txt

COPY vneuron /vneuron

WORKDIR /vneuron

EXPOSE 5000

CMD ["python3" , "./flask_deploy/app.py"]
