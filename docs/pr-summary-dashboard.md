# PR Summary Dashboard Support

This document outlines the dashboard support that has been added for PR summary functionality.

## Overview

The PR summary feature now includes comprehensive dashboard support for tracking and analyzing PR summary generation across repositories.

## What Was Added

### 1. Database Model (`app/models/analytics.py`)

Added `PRSummary` model to track:
- PR details (title, body length, files changed)
- Summary generation metrics (success, length, type)
- RAG system usage (available, generated vs fallback)
- Response tracking (posted, response time)

### 2. Analytics Service (`app/services/analytics_service.py`)

Added `log_pr_summary()` method to log detailed PR summary statistics including:
- PR metadata (title, body length, files)
- Summary generation details (type, length, success)
- RAG system availability and usage
- Response posting status

Updated `get_repository_stats()` to include PR summary statistics:
- Total summaries generated
- Success rates
- RAG vs fallback usage
- Average summary length

### 3. RAG Service Integration (`app/services/rag_service.py`)

Updated `handle_pr_opened_event()` to:
- Track summary generation process
- Log detailed metrics to analytics service
- Distinguish between RAG-generated and fallback summaries
- Monitor success rates and response times

### 4. Dashboard API Endpoints (`app/api/dashboard.py`)

#### Updated Models
- `RepositoryStats`: Added `pr_summaries_generated` field

#### New Endpoints

1. **Repository PR Summary Stats**
   ```
   GET /repositories/{owner}/{repo}/pr-summaries
   ```
   Returns detailed PR summary statistics for a specific repository:
   - Total summaries generated
   - Success rates
   - RAG usage statistics
   - Average summary length

2. **PR Summary Analytics**
   ```
   GET /analytics/pr-summaries
   ```
   Returns comprehensive PR summary analytics:
   - Summary generation trends
   - RAG vs fallback usage rates
   - Timeline data for PR summary actions
   - Performance metrics

## Database Migration Required

To use the new PR summary dashboard features, you need to run the database migration:

```bash
python -m alembic revision --autogenerate -m "Add PRSummary table"
python -m alembic upgrade head
```

## Dashboard Metrics Available

### Repository Level
- **Total PR Summaries**: Number of summaries generated
- **Success Rate**: Percentage of successful summary generations
- **RAG Usage Rate**: Percentage of summaries generated using RAG vs fallback
- **Average Summary Length**: Average character count of generated summaries

### System Level
- **PR Summary Trends**: Timeline of summary generation activity
- **Performance Metrics**: Response times and success rates
- **RAG Effectiveness**: Comparison of RAG-generated vs fallback summaries

## Usage Examples

### Get Repository PR Summary Stats
```bash
curl -H "Authorization: Bearer YOUR_ADMIN_TOKEN" \
  "http://localhost:8000/api/dashboard/repositories/owner/repo/pr-summaries"
```

### Get PR Summary Analytics
```bash
curl -H "Authorization: Bearer YOUR_ADMIN_TOKEN" \
  "http://localhost:8000/api/dashboard/analytics/pr-summaries?days=30"
```

## Integration with Existing Dashboard

The PR summary metrics are now integrated into:
- Repository overview statistics
- Activity timeline (shows `pr_summary` action type)
- Recent actions feed
- Performance analytics

## Monitoring and Alerts

The system now tracks:
- Failed summary generations
- RAG system availability
- Response time degradation
- Success rate trends

This enables proactive monitoring of PR summary functionality and helps identify issues with the RAG system or summary generation process. 