FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && apt-get clean

WORKDIR /app

COPY artifacts /app/artifacts
COPY reporting /app/reporting

COPY serving/api.py /app/
COPY serving/requirements.txt /app/

RUN pip install --upgrade pip
RUN pip install --timeout=2000 -r requirements.txt

EXPOSE 8080

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080", "--reload"]