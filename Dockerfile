FROM python:3.13

WORKDIR /code


# Install PyTorch + CUDA 12.8 builds
RUN pip install --no-cache-dir \
torch==2.8.0+cu128 \
torchvision==0.23.0+cu128 \
torchaudio==2.8.0+cu128 \
--extra-index-url https://download.pytorch.org/whl/cu128

COPY ./requirements.txt /code/requirements.txt

#install requirements.txt
RUN pip install --no-cache-dir -r /code/requirements.txt



COPY ./Model /code/Model

COPY ./Tokenizer /code/Tokenizer

COPY ./bert_architecture.py /code/bert_architecture.py

COPY ./Training_Data /code/Training_Data

COPY ./server.py /code/server.py

EXPOSE 8000

CMD ["python","-m","uvicorn","server:app", "--host" , "0.0.0.0" , "--port" , "8000"]