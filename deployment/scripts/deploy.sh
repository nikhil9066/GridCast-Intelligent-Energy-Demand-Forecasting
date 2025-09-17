#!/bin/bash

# GridCast Deployment Script
# This script handles the deployment of GridCast API to various environments

set -e  # Exit on error

# Configuration
PROJECT_NAME="gridcast"
IMAGE_NAME="gridcast-api"
CONTAINER_NAME="gridcast-container"
API_PORT=5000
DOCKER_COMPOSE_FILE="deployment/docker/docker-compose.yml"

# Colors for output
RED='\\033[0;31m'
GREEN='\\033[0;32m'
YELLOW='\\033[1;33m'
BLUE='\\033[0;34m'
NC='\\033[0m' # No Color

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_dependencies() {
    log_info "Checking dependencies..."

    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi

    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi

    # Check if Docker is running
    if ! docker info &> /dev/null; then
        log_error "Docker is not running. Please start Docker first."
        exit 1
    fi

    log_success "All dependencies are available"
}

build_image() {
    log_info "Building Docker image..."

    # Build the Docker image
    docker build -f deployment/docker/Dockerfile -t $IMAGE_NAME:latest .

    if [ $? -eq 0 ]; then
        log_success "Docker image built successfully"
    else
        log_error "Failed to build Docker image"
        exit 1
    fi
}

stop_existing() {
    log_info "Stopping existing containers..."

    # Stop existing container if running
    if [ "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
        docker stop $CONTAINER_NAME
        log_success "Stopped existing container"
    fi

    # Remove existing container
    if [ "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
        docker rm $CONTAINER_NAME
        log_success "Removed existing container"
    fi
}

deploy_standalone() {
    log_info "Deploying standalone container..."

    # Run the container
    docker run -d \\
        --name $CONTAINER_NAME \\
        -p $API_PORT:5000 \\
        -v "$(pwd)/models:/app/models" \\
        -v "$(pwd)/data:/app/data" \\
        -v "$(pwd)/logs:/app/logs" \\
        --restart unless-stopped \\
        $IMAGE_NAME:latest

    if [ $? -eq 0 ]; then
        log_success "Container deployed successfully"
        log_info "API available at: http://localhost:$API_PORT"
    else
        log_error "Failed to deploy container"
        exit 1
    fi
}

deploy_compose() {
    log_info "Deploying with Docker Compose..."

    # Use Docker Compose
    docker-compose -f $DOCKER_COMPOSE_FILE up -d

    if [ $? -eq 0 ]; then
        log_success "Docker Compose deployment successful"
        log_info "API available at: http://localhost:$API_PORT"
    else
        log_error "Docker Compose deployment failed"
        exit 1
    fi
}

health_check() {
    log_info "Performing health check..."

    # Wait for container to start
    sleep 10

    # Check health endpoint
    max_attempts=30
    attempt=1

    while [ $attempt -le $max_attempts ]; do
        if curl -f -s "http://localhost:$API_PORT/health" &> /dev/null; then
            log_success "Health check passed"
            return 0
        fi

        log_info "Attempt $attempt/$max_attempts: Waiting for API to be ready..."
        sleep 5
        ((attempt++))
    done

    log_error "Health check failed - API is not responding"
    return 1
}

show_status() {
    log_info "Container status:"
    docker ps -f name=$CONTAINER_NAME

    echo ""
    log_info "Recent logs:"
    docker logs --tail 20 $CONTAINER_NAME

    echo ""
    log_info "API endpoints:"
    echo "  Health check: http://localhost:$API_PORT/health"
    echo "  Documentation: http://localhost:$API_PORT/"
    echo "  Forecast: POST http://localhost:$API_PORT/forecast"
}

cleanup() {
    log_info "Cleaning up..."

    # Stop and remove containers
    docker-compose -f $DOCKER_COMPOSE_FILE down 2>/dev/null || true
    docker stop $CONTAINER_NAME 2>/dev/null || true
    docker rm $CONTAINER_NAME 2>/dev/null || true

    # Remove images
    docker rmi $IMAGE_NAME:latest 2>/dev/null || true

    log_success "Cleanup completed"
}

show_help() {
    echo "GridCast Deployment Script"
    echo ""
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  build              Build Docker image"
    echo "  deploy             Deploy using standalone Docker container"
    echo "  deploy-compose     Deploy using Docker Compose"
    echo "  stop               Stop running containers"
    echo "  restart            Restart the deployment"
    echo "  status             Show deployment status"
    echo "  logs               Show container logs"
    echo "  health             Check API health"
    echo "  cleanup            Clean up all containers and images"
    echo "  help               Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 deploy          # Build and deploy standalone container"
    echo "  $0 deploy-compose  # Deploy with Docker Compose"
    echo "  $0 restart         # Restart the deployment"
}

# Main script logic
case "${1:-help}" in
    build)
        check_dependencies
        build_image
        ;;

    deploy)
        check_dependencies
        build_image
        stop_existing
        deploy_standalone
        health_check
        show_status
        ;;

    deploy-compose)
        check_dependencies
        deploy_compose
        health_check
        show_status
        ;;

    stop)
        log_info "Stopping GridCast deployment..."
        docker-compose -f $DOCKER_COMPOSE_FILE down 2>/dev/null || true
        docker stop $CONTAINER_NAME 2>/dev/null || true
        log_success "Deployment stopped"
        ;;

    restart)
        log_info "Restarting GridCast deployment..."
        $0 stop
        sleep 5
        $0 deploy
        ;;

    status)
        show_status
        ;;

    logs)
        log_info "Showing container logs:"
        docker logs -f $CONTAINER_NAME
        ;;

    health)
        health_check
        ;;

    cleanup)
        cleanup
        ;;

    help|--help|-h)
        show_help
        ;;

    *)
        log_error "Unknown command: $1"
        show_help
        exit 1
        ;;
esac