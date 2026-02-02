FROM python:3.11

WORKDIR /app

COPY requirements.txt .
COPY install_torch.sh .

RUN pip install -r requirements.txt
RUN sh install_torch.sh

COPY . .

ENTRYPOINT [ "python3", "-m" ,"flask", "run", "--host=0.0.0.0", "--port=5000" ]