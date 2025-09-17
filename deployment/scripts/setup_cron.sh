#!/bin/bash

# GridCast Automated Retraining Setup Script
# This script sets up automated retraining via cron jobs

set -e

# Configuration
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RETRAIN_SCRIPT="$PROJECT_DIR/deployment/scripts/retrain_pipeline.py"
LOG_DIR="$PROJECT_DIR/logs"
PYTHON_ENV="${PYTHON_ENV:-python3}"

# Colors
GREEN='\\033[0;32m'
YELLOW='\\033[1;33m'
BLUE='\\033[0;34m'
NC='\\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

create_log_dir() {
    log_info "Creating log directory..."
    mkdir -p "$LOG_DIR"
    log_success "Log directory created: $LOG_DIR"
}

setup_weekly_retrain() {
    log_info "Setting up weekly retraining cron job..."

    # Create cron job entry
    CRON_JOB="0 2 * * 0 cd $PROJECT_DIR && $PYTHON_ENV $RETRAIN_SCRIPT >> $LOG_DIR/weekly_retrain.log 2>&1"

    # Add to crontab
    (crontab -l 2>/dev/null | grep -v "$RETRAIN_SCRIPT"; echo "$CRON_JOB") | crontab -

    log_success "Weekly retraining scheduled for Sundays at 2:00 AM"
}

setup_monthly_retrain() {
    log_info "Setting up monthly retraining cron job..."

    # Create cron job entry (1st of each month at 3:00 AM)
    CRON_JOB="0 3 1 * * cd $PROJECT_DIR && $PYTHON_ENV $RETRAIN_SCRIPT --force >> $LOG_DIR/monthly_retrain.log 2>&1"

    # Add to crontab
    (crontab -l 2>/dev/null | grep -v "$RETRAIN_SCRIPT --force"; echo "$CRON_JOB") | crontab -

    log_success "Monthly retraining scheduled for 1st of each month at 3:00 AM"
}

setup_health_monitoring() {
    log_info "Setting up API health monitoring..."

    # Create health check script
    HEALTH_SCRIPT="$PROJECT_DIR/deployment/scripts/health_check.sh"

    cat > "$HEALTH_SCRIPT" << 'EOF'
#!/bin/bash

# GridCast API Health Check
API_URL="http://localhost:5000/health"
LOG_FILE="logs/health_check.log"

# Create log directory if it doesn't exist
mkdir -p "$(dirname "$LOG_FILE")"

# Check API health
response=$(curl -s -w "%{http_code}" -o /dev/null "$API_URL" 2>/dev/null || echo "000")

timestamp=$(date '+%Y-%m-%d %H:%M:%S')

if [ "$response" = "200" ]; then
    echo "[$timestamp] API Health: OK" >> "$LOG_FILE"
else
    echo "[$timestamp] API Health: FAILED (HTTP $response)" >> "$LOG_FILE"

    # Send alert (placeholder - implement actual alerting)
    echo "[$timestamp] ALERT: GridCast API is down!" >> "$LOG_FILE"
fi
EOF

    chmod +x "$HEALTH_SCRIPT"

    # Setup health check every 5 minutes
    HEALTH_CRON="*/5 * * * * cd $PROJECT_DIR && $HEALTH_SCRIPT"
    (crontab -l 2>/dev/null | grep -v "health_check.sh"; echo "$HEALTH_CRON") | crontab -

    log_success "Health monitoring setup (checks every 5 minutes)"
}

create_config_file() {
    log_info "Creating retraining configuration file..."

    CONFIG_FILE="$PROJECT_DIR/deployment/scripts/retrain_config.json"

    cat > "$CONFIG_FILE" << EOF
{
    "data_source": "file",
    "retrain_threshold_days": 7,
    "performance_threshold": 0.03,
    "max_models_to_keep": 5,
    "notification_email": null,
    "backup_models": true,
    "validate_before_deploy": true,
    "mlflow_uri": "file:../../models/mlflow_artifacts",
    "model_dir": "../../models/saved_models",
    "data_dir": "../../data"
}
EOF

    log_success "Configuration file created: $CONFIG_FILE"
}

show_cron_status() {
    log_info "Current cron jobs:"
    crontab -l | grep -E "(retrain_pipeline|health_check)" || echo "No GridCast cron jobs found"
}

remove_cron_jobs() {
    log_info "Removing GridCast cron jobs..."

    # Remove cron jobs
    crontab -l 2>/dev/null | grep -v "$RETRAIN_SCRIPT" | grep -v "health_check.sh" | crontab -

    log_success "GridCast cron jobs removed"
}

show_help() {
    echo "GridCast Automated Retraining Setup"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  setup-weekly     Setup weekly retraining (Sundays 2 AM)"
    echo "  setup-monthly    Setup monthly retraining (1st of month 3 AM)"
    echo "  setup-monitoring Setup API health monitoring (every 5 minutes)"
    echo "  setup-all        Setup all automation"
    echo "  status           Show current cron job status"
    echo "  remove           Remove all GridCast cron jobs"
    echo "  help             Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 setup-all      # Setup complete automation"
    echo "  $0 status         # Check current cron jobs"
}

# Main script logic
case "${1:-help}" in
    setup-weekly)
        create_log_dir
        create_config_file
        setup_weekly_retrain
        show_cron_status
        ;;

    setup-monthly)
        create_log_dir
        create_config_file
        setup_monthly_retrain
        show_cron_status
        ;;

    setup-monitoring)
        create_log_dir
        setup_health_monitoring
        show_cron_status
        ;;

    setup-all)
        create_log_dir
        create_config_file
        setup_weekly_retrain
        setup_monthly_retrain
        setup_health_monitoring
        show_cron_status
        log_success "Complete automation setup finished!"
        ;;

    status)
        show_cron_status
        ;;

    remove)
        remove_cron_jobs
        ;;

    help|--help|-h)
        show_help
        ;;

    *)
        echo "Unknown command: $1"
        show_help
        exit 1
        ;;
esac