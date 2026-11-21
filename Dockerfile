
FROM python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90

WORKDIR /app

COPY selector_v4.py /app/selector_v4.py
COPY selector_v3_candidate.py /app/selector_v3_candidate.py
COPY fixture_map.json /app/fixture_map.json
COPY line_map.json /app/line_map.json

USER nobody

CMD ["python", "/app/selector_v4.py"]
