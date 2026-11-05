# The web app in a container, for hosting on an institute server or research platform.
#
#   docker build -t doekit .
#   docker run -p 8501:8501 doekit      then open http://localhost:8501
#
# Only the code, manual, templates and simulated examples go into the image. The real data
# in archive/ and data/ stays out (see .dockerignore), and the app saves nothing on the server.

FROM python:3.12-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY app ./app
COPY configs ./configs
COPY docs ./docs
COPY .streamlit ./.streamlit
RUN pip install .

# Some research platforms run containers under an arbitrary user ID, not as root,
# so nothing here relies on a particular user.
RUN chmod -R g=u /app
USER 1001

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
