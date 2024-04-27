# app/Dockerfile

FROM python:3.10-slim

ARG POETRY_VERSION=1.7.1

RUN apt-get update && apt-get install -y \
	build-essential \
	curl \
	software-properties-common \
	git \
	&& rm -rf /var/lib/apt/lists/*

# COPY . .  
RUN git clone https://github.com/jeremyforest/eduLLM.git

WORKDIR /eduLLM
# ENV PYTHONPATH=${PYTHONPATH}:${PWD}

# Poetry install
# TODO Fix the dir copy for that. Right now copied manually in the docker folder which is bad
RUN pip install poetry==${POETRY_VERSION}

# Project init via POETRY
# RUN poetry config virtualenvs.create false
# RUN poetry install
# Their is a bug with poetry config virtualenvs.create false https://github.com/python-poetry/poetry/issues/6459
# Below is a workaround to set poetry env as default env
RUN poetry install --no-interaction --no-ansi
RUN POETRY_ENV_PATH=$(poetry env info --path) && ln -s $POETRY_ENV_PATH /mypyenv
ENV PATH="/mypyenv/bin:$PATH"

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["poetry", "run", "streamlit", "run", "edullm/ui/ui.py"]
