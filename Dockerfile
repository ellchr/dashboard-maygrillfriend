FROM ghcr.io/huggingface/spaces/streamlit-blank:main

# Salin file requirements dan instal dependensi
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

# Salin seluruh sisa kodingan ke dalam server
COPY . /app
