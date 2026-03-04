web: gunicorn --bind=0.0.0.0:8000 --workers=4 --forwarded-allow-ips=* --timeout=120 --graceful-timeout=30 app:app
worker: python -u worker.py
