module.exports = {
  apps: [
    {
      name: "gitbot",
      script: "uvicorn",
      args: "app.main:app --host 0.0.0.0 --port 8000",
      interpreter: "venv/bin/python",
      watch: false,
      env: {
        PYTHONUNBUFFERED: "1"
      }
    }
  ]
}; 