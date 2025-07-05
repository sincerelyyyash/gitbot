# GitBot Troubleshooting Guide

This guide helps resolve common issues with GitBot, especially those related to Google API configuration.

## Quick Status Check

Use the health endpoint to check GitBot's current status:
```bash
curl http://your-server:8050/health
```

## Common Issues and Solutions

### 1. Google API Key IP Address Restrictions

**Error Message:**
```
403 The provided API key has an IP address restriction. 
The originating IP address of the call (X.X.X.X) violates this restriction.
```

**What This Means:**
Your Google Gemini API key has been configured with IP address restrictions in Google Cloud Console, and the current server IP is not on the allowed list.

**Solutions:**

#### Option A: Remove IP Restrictions (Recommended for Development)
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Navigate to **APIs & Services > Credentials**
3. Find your API key and click on it
4. In the **API restrictions** section, find **IP addresses**
5. Remove all IP restrictions or change to "None"
6. Click **Save**

#### Option B: Add Server IP to Allowed List
1. Find your server's public IP address:
   ```bash
   curl ifconfig.me
   ```
2. Go to [Google Cloud Console](https://console.cloud.google.com/)
3. Navigate to **APIs & Services > Credentials**
4. Find your API key and click on it
5. In the **API restrictions** section, find **IP addresses**
6. Add your server's IP address to the allowed list
7. Click **Save**

#### Option C: Use a Different API Key
Create a new API key without IP restrictions:
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Navigate to **APIs & Services > Credentials**
3. Click **Create Credentials > API Key**
4. Don't add any IP restrictions
5. Update your `.env` file with the new key

### 2. Google API Quota Exceeded

**Error Message:**
```
Quota exceeded / Resource exhausted
```

**Solutions:**
1. **Wait for Reset:** Quotas typically reset daily
2. **Check Usage:** Go to Google Cloud Console > APIs & Services > Quotas
3. **Increase Limits:** Request quota increases if needed
4. **Optimize Usage:** Reduce the frequency of requests

### 3. Invalid API Key

**Error Message:**
```
Invalid API key / Unauthorized
```

**Solutions:**
1. **Verify Key:** Check that your `GEMINI_API_KEY` in `.env` is correct
2. **Check Permissions:** Ensure the API key has access to Generative AI APIs
3. **Generate New Key:** Create a fresh API key if the current one is corrupted

### 4. Network Connectivity Issues

**Error Message:**
```
Network error / Connection timeout
```

**Solutions:**
1. **Check Internet:** Verify server has internet access
2. **Firewall Settings:** Ensure outbound HTTPS traffic is allowed
3. **DNS Resolution:** Test if `generativelanguage.googleapis.com` is accessible
4. **Proxy Settings:** Check if proxy configuration is needed

## Configuration Validation

### Environment Variables Checklist

Ensure these variables are set in your `.env` file:
```bash
# Required for basic functionality
GITHUB_APP_ID=your_app_id
GITHUB_PRIVATE_KEY="-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----"
GITHUB_WEBHOOK_SECRET=your_webhook_secret

# Required for RAG functionality
GEMINI_API_KEY=your_gemini_api_key

# Optional
LOG_LEVEL=INFO
ENVIRONMENT=production
```

### API Key Validation Test

Test your API key manually:
```bash
curl -H "Content-Type: application/json" \
     -d '{"model":"models/embedding-001","content":{"parts":[{"text":"test"}]}}' \
     -H "x-goog-api-key: YOUR_API_KEY" \
     https://generativelanguage.googleapis.com/v1beta/models/embedding-001:embedContent
```

## Monitoring and Logs

### Application Logs

Check GitBot logs for detailed error information:
```bash
# If running with Docker
docker logs gitbot-container

# If running directly
tail -f logs/gitbot.log
```

### Health Monitoring

Set up monitoring for the health endpoint:
```bash
# Simple health check
curl http://your-server:8050/health

# Detailed status with jq
curl -s http://your-server:8050/health | jq .
```

## Fallback Behavior

When RAG functionality is unavailable, GitBot will:
- ✅ Still respond to GitHub events
- ✅ Provide helpful error messages to users
- ✅ Give specific guidance based on error type
- ✅ Continue normal webhook processing

Users will see messages like:
- **IP Restrictions:** "Configuration issue - contact administrator"
- **Quota Exceeded:** "Quota limit reached - try again tomorrow"
- **Invalid Key:** "Configuration issue - contact administrator"

## Getting Help

If issues persist:

1. **Check Logs:** Look for detailed error messages in application logs
2. **Test Configuration:** Use the validation endpoints
3. **Verify Network:** Ensure connectivity to Google APIs
4. **Update Documentation:** This troubleshooting guide
5. **Contact Support:** Provide logs and error details

## Prevention

### Best Practices
- Monitor API usage and quotas
- Set up alerts for quota thresholds
- Use health checks in production
- Keep API keys secure but accessible
- Document any custom configurations

### Regular Maintenance
- Monitor disk space for ChromaDB persistence
- Check for inactive repository collections
- Review API usage patterns
- Update dependencies regularly 