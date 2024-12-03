web: gunicorn -w 4 -k uvicorn.workers.UvicornWorker app:app
worker: python -m src.core.background_tasks
quantum_worker: python -m src.quantum.quantum_processing
monitoring: python -m src.monitoring.system_monitor
