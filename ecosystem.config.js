module.exports = {
  apps: [
    {
      name: "gitbot",
      script: "/root/venv/bin/uvicorn",
      args: "app.main:app --host 0.0.0.0 --port 8050",
      interpreter: "/root/venv/bin/python",
      watch: false,
      env: {
        PYTHONUNBUFFERED: "1"
      }
    }
  ]
}; 