# Blast Radius selector. Same base image (by digest) as the environment the tests were
# labelled in, so any sqlglot test it runs behaves as it did there.
FROM python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_ROOT_USER_ACTION=ignore
COPY requirements.lock /app/requirements.lock
RUN pip install -q -r /app/requirements.lock
# sqlglot at the pinned commit, from its own GitHub repository; the build fails on any
# content mismatch.
COPY build/fetch_sqlglot.py /app/build/fetch_sqlglot.py
RUN python /app/build/fetch_sqlglot.py /app/sqlglot \
 && python -m compileall -q /app/sqlglot/sqlglot /app/sqlglot/tests
COPY selector/ /app/selector/
COPY map/ /app/map/
# identifier reference index of the pinned checkout (build CPU is not charged)
RUN python /app/selector/refindex.py /app/sqlglot /app/map/refs.json \
 && python -m compileall -q /app/selector && chmod -R a+rX /app
ENV BR_DATA=/data BR_REPO=/app/sqlglot BR_MAP=/app/map/blast.map BR_REFS=/app/map/refs.json \
    BR_WORK=/tmp/br PYTHONDONTWRITEBYTECODE=1
CMD ["python", "/app/selector/main.py"]
