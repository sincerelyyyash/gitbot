module.exports = {
  apps: [
    {
      name: "gitbot",
      script: "../venv/bin/uvicorn",
      args: "app.main:app --host 0.0.0.0 --port 8050",
      interpreter: "../venv/bin/python",
      watch: false,
      env: {
        PYTHONUNBUFFERED: "1"
      }
    }
  ]
}; 