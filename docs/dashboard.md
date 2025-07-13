# GitBot Dashboard & Analytics

The GitBot dashboard provides comprehensive analytics and monitoring for your GitHub bot's activities, performance, and system health.

## Features

### 📊 **Analytics Dashboard**
- **Repository Statistics**: Track actions, success rates, and performance per repository
- **Activity Timeline**: Visualize bot activity over time with interactive charts
- **Performance Metrics**: Monitor response times, success rates, and error patterns
- **System Health**: Real-time monitoring of all system components

### 🔍 **Action Tracking**
GitBot automatically tracks all actions including:
- **Issue Comments**: Questions answered, analysis provided
- **Issue Analysis**: Similarity detection, duplicate identification
- **Pull Request Reviews**: Security scans, code quality analysis
- **Repository Indexing**: Content processing, knowledge base updates

### 📈 **Key Metrics**
- Total actions performed (daily/weekly/monthly)
- Success rates by action type
- Average response times
- API usage and quota consumption
- Repository activity levels
- Error rates and patterns

## API Endpoints

All dashboard endpoints require admin authentication via the `admin_token` parameter.

### Overview
```bash
GET /api/dashboard/overview?admin_token=your_token
```
Returns high-level metrics and system status.

### Repository Analytics
```bash
# Get all repositories
GET /api/dashboard/repositories?admin_token=your_token

# Get specific repository stats
GET /api/dashboard/repositories/{owner}/{repo}?admin_token=your_token
```

### Activity Timeline
```bash
GET /api/dashboard/activity/timeline?days=30&repo_full_name=owner/repo&admin_token=your_token
```

### Recent Actions
```bash
GET /api/dashboard/activity/recent?limit=50&repo_full_name=owner/repo&admin_token=your_token
```

### System Health
```bash
GET /api/dashboard/health?admin_token=your_token
```

### Analytics Summary
```bash
GET /api/dashboard/analytics/summary?days=30&repo_full_name=owner/repo&admin_token=your_token
```

### Performance Metrics
```bash
GET /api/dashboard/analytics/performance?days=7&repo_full_name=owner/repo&admin_token=your_token
```

### Data Export
```bash
GET /api/dashboard/export/actions?format=json&start_date=2024-01-01&end_date=2024-01-31&admin_token=your_token
```

## Database Schema

### Core Tables

#### `repositories`
Tracks registered repositories and their metadata.

#### `action_logs`
Central log of all GitBot actions with timing and status information.

#### `issue_analyses`
Detailed tracking of issue analysis results including similarity detection.

#### `pr_analyses`
Pull request analysis results including security, quality, and complexity metrics.

#### `indexing_jobs`
Repository indexing job status and performance metrics.

#### `usage_metrics`
Daily aggregated usage statistics and API consumption.

#### `system_health`
System component health check results.

## Setup and Configuration

### Database Configuration

By default, GitBot uses SQLite for analytics data:
```bash
DATABASE_URL=sqlite+aiosqlite:///./data/gitbot.db
```

For production, use PostgreSQL:
```bash
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/gitbot
```

### Docker Compose Setup

The provided `docker-compose.yml` includes:
- PostgreSQL database for analytics
- GitBot application with database integration
- ChromaDB for vector storage

```bash
# Start with PostgreSQL
docker-compose up -d

# Or use SQLite (comment out postgres dependency in docker-compose.yml)
docker-compose up -d app chromadb
```

### Environment Variables

```bash
# Required for dashboard access
ADMIN_TOKEN=your_secure_admin_token

# Database configuration
DATABASE_URL=postgresql+asyncpg://gitbot:gitbot_password@postgres:5432/gitbot
DATABASE_ECHO=false  # Set to true for SQL debugging

# Other GitBot configuration
GITHUB_APP_ID=your_app_id
GITHUB_PRIVATE_KEY="-----BEGIN PRIVATE KEY-----..."
GITHUB_WEBHOOK_SECRET=your_webhook_secret
GEMINI_API_KEY=your_gemini_api_key
```

### Database Migrations

Initialize the database schema:
```bash
# Install dependencies
pip install -r requirements.txt

# Run migrations
alembic upgrade head
```

## Monitoring and Alerts

### Health Checks

The dashboard provides comprehensive health monitoring:

- **Database Connectivity**: Connection status and query performance
- **GitHub API**: Rate limits and authentication status
- **Gemini API**: Quota usage and response times
- **ChromaDB**: Vector database connectivity

### Performance Monitoring

Track key performance indicators:

- **Response Time**: P50, P95, P99 percentiles by action type
- **Success Rate**: Percentage of successful actions
- **Error Patterns**: Common failure modes and frequencies
- **Resource Usage**: API quota consumption and database performance

### Alerting

Set up monitoring alerts based on:
- Success rate drops below threshold (e.g., <90%)
- Response time increases significantly
- Error rate spikes
- API quota approaching limits
- System health degradation

## Usage Examples

### Dashboard Overview
```json
{
  "total_repositories": 25,
  "active_repositories": 18,
  "total_actions_today": 147,
  "total_actions_this_week": 892,
  "success_rate_today": 0.94,
  "avg_response_time_ms": 2340,
  "system_health": "healthy",
  "last_updated": "2024-01-15T10:30:00Z"
}
```

### Repository Statistics
```json
{
  "repo_full_name": "owner/repo",
  "total_actions": 456,
  "recent_activity_count": 23,
  "success_rate": 0.96,
  "avg_response_time_ms": 1850,
  "issues_analyzed": 89,
  "prs_analyzed": 34,
  "duplicates_found": 12,
  "security_issues_found": 5,
  "last_activity": "2024-01-15T09:45:00Z",
  "indexed_at": "2024-01-14T15:20:00Z"
}
```

### Activity Timeline
```json
[
  {
    "date": "2024-01-15",
    "action_type": "issue_comment",
    "count": 23,
    "successful": 22,
    "success_rate": 0.96
  },
  {
    "date": "2024-01-15",
    "action_type": "pr_analysis",
    "count": 8,
    "successful": 8,
    "success_rate": 1.0
  }
]
```

## Best Practices

### Data Retention
- Configure automatic cleanup of old action logs
- Archive detailed analytics data after 90 days
- Maintain summary metrics for historical analysis

### Performance Optimization
- Use database indexes for common query patterns
- Implement connection pooling for high-traffic scenarios
- Cache frequently accessed metrics

### Security
- Protect admin endpoints with strong authentication
- Use HTTPS in production
- Regularly rotate admin tokens
- Audit dashboard access logs

### Monitoring
- Set up automated health checks
- Monitor database performance and growth
- Track API quota usage trends
- Alert on system degradation

## Troubleshooting

### Common Issues

**Database Connection Errors**
```bash
# Check database status
docker-compose ps postgres

# View logs
docker-compose logs postgres
docker-compose logs app
```

**Missing Analytics Data**
- Verify tracking integration is enabled
- Check for database migration issues
- Ensure proper error handling in action handlers

**Performance Issues**
- Monitor database query performance
- Check for missing indexes
- Analyze slow query logs

**Dashboard Access Issues**
- Verify admin token configuration
- Check CORS settings
- Ensure proper authentication headers

For more help, check the main GitBot documentation or open an issue on GitHub. 