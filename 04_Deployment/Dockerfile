FROM python:3.14-slim-bookworm

WORKDIR /app

# install pip
RUN pip install pipenv

# copy dependency files first(for Docker caching)
COPY Pipfile Pipfile.lock ./

# install dependencies into container python
RUN pipenv install --system --deploy

# copy application source code
COPY predict.py transformers.py model_C=1.bin ./

# Expose application port
EXPOSE 9696

# start the application
CMD ["waitress-serve", "--listen=0.0.0.0:9696", "predict:app"]


