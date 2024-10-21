# syntax=docker/dockerfile:1

FROM continuumio/anaconda3:2024.06-1
WORKDIR /app
COPY requirements2.txt .
RUN conda config --append channels conda-forge
RUN conda install --file ./requirements2.txt
RUN pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
RUN pip install torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu
RUN pip install -U ray
RUN pip install -U "ray[data,train,tune,serve]"
COPY . .
WORKDIR /app/src
# TODO: one for server, one for client (with seed arg)
# CMD ["python", "main.py --fl_mode server"]
# CMD ["ls"]
# CMD ["python", "src/main.py --fl_mode client --seed 1"]
EXPOSE 3000
EXPOSE 8080