web: gunicorn --workers=2 --bind=0.0.0.0:${PORT:-8000} --forwarded-allow-ips=* --timeout=120 --graceful-timeout=30 app:app
worker: python -u worker.py
