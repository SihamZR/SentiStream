FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6 && \
    apt-get clean

WORKDIR /app

COPY streamlit-app.py .
COPY requirements.txt .

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

EXPOSE 8081

CMD ["streamlit", "run", "streamlit-app.py", "--server.port=8081", "--server.address=0.0.0.0"]