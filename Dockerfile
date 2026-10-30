# Read-only public demo: shows the audit, experiment and shortlist. No keys, no credits spent.
FROM python:3.12-slim
WORKDIR /app
COPY requirements-demo.txt .
RUN pip install --no-cache-dir -r requirements-demo.txt
COPY . .
ENV CATHODEPILOT_PUBLIC=1
EXPOSE 7860
CMD ["python", "-m", "uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "7860"]
