#!/bin/bash
# RTAI Trading System - Production Deployment Script
# One-command deployment with health checks and rollback capability

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="rtai-trading"
TARGET_IMAGE_SIZE_MB=190
HEALTH_TIMEOUT=60

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
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

# Pre-deployment checks
check_prerequisites() {
    log_info "🔍 Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed"
        exit 1
    fi
    
    # Check .env file
    if [[ ! -f "$SCRIPT_DIR/.env" ]]; then
        log_warning ".env file not found"
        log_info "Creating .env from .env.example..."
        cp "$SCRIPT_DIR/.env.example" "$SCRIPT_DIR/.env"
        log_warning "Please edit .env file with your configuration"
        exit 1
    fi
    
    # Verify no matplotlib imports
    log_info "🧟 Checking for zombie files..."
    if grep -r "import matplotlib\|from matplotlib" "$SCRIPT_DIR/rtai/" 2>/dev/null; then
        log_error "❌ ZOMBIE DETECTED: matplotlib imports found in rtai/"
        exit 1
    fi
    
    log_success "✅ Prerequisites check passed"
}

# Build optimized Docker image
build_image() {
    log_info "🔨 Building optimized Docker image..."
    
    # Build with build args
    docker build \
        --target runtime \
        --tag "${PROJECT_NAME}:latest" \
        --tag "${PROJECT_NAME}:$(date +%Y%m%d-%H%M%S)" \
        "$SCRIPT_DIR"
    
    # Check image size
    IMAGE_SIZE_BYTES=$(docker images "${PROJECT_NAME}:latest" --format "table {{.Size}}" | tail -n +2 | head -1)
    IMAGE_SIZE_MB=$(docker images "${PROJECT_NAME}:latest" --format "table {{.Size}}" | tail -n +2 | head -1 | sed 's/MB//')
    
    log_info "📦 Image size: ${IMAGE_SIZE_BYTES}"
    
    # Validate image size (approximate check)
    if [[ "${IMAGE_SIZE_BYTES}" == *"GB"* ]]; then
        log_error "❌ Image size exceeds target (<${TARGET_IMAGE_SIZE_MB}MB)"
        exit 1
    fi
    
    log_success "✅ Docker image built successfully"
}

# Deploy services
deploy_services() {
    log_info "🚀 Deploying services..."
    
    # Stop existing containers
    docker-compose down --remove-orphans || true
    
    # Start services
    docker-compose up -d
    
    log_success "✅ Services deployed"
}

# Health check
wait_for_health() {
    log_info "🏥 Waiting for health check..."
    
    local attempts=0
    local max_attempts=$((HEALTH_TIMEOUT / 5))
    
    while [[ $attempts -lt $max_attempts ]]; do
        if curl -s -f http://localhost:8080/health > /dev/null 2>&1; then
            log_success "✅ Health check passed"
            return 0
        fi
        
        log_info "⏳ Waiting for service to be healthy... (attempt $((attempts + 1))/${max_attempts})"
        sleep 5
        ((attempts++))
    done
    
    log_error "❌ Health check failed after ${HEALTH_TIMEOUT}s"
    return 1
}

# Show service status
show_status() {
    log_info "📊 Service Status:"
    docker-compose ps
    
    echo ""
    log_info "🏥 Health Status:"
    curl -s http://localhost:8080/health/status | python -m json.tool || echo "Health endpoint not available"
    
    echo ""
    log_info "📈 Available Endpoints:"
    echo "  Health: http://localhost:8080/health"
    echo "  Status: http://localhost:8080/health/status"
    echo "  Metrics: http://localhost:8080/metrics"
    
    echo ""
    log_info "📋 Logs:"
    echo "  View logs: docker-compose logs -f rtai"
    echo "  View metrics: curl http://localhost:8080/metrics"
}

# Rollback function
rollback() {
    log_warning "🔄 Rolling back deployment..."
    
    # Stop current deployment
    docker-compose down
    
    # Find previous image
    PREVIOUS_IMAGE=$(docker images "${PROJECT_NAME}" --format "table {{.Tag}}" | grep -E "[0-9]{8}-[0-9]{6}" | head -2 | tail -1)
    
    if [[ -n "$PREVIOUS_IMAGE" ]]; then
        log_info "Rolling back to ${PROJECT_NAME}:${PREVIOUS_IMAGE}"
        docker tag "${PROJECT_NAME}:${PREVIOUS_IMAGE}" "${PROJECT_NAME}:latest"
        docker-compose up -d
        log_success "✅ Rollback completed"
    else
        log_error "❌ No previous image found for rollback"
        exit 1
    fi
}

# Main deployment function
main() {
    log_info "🎯 Starting RTAI Trading System deployment..."
    
    case "${1:-deploy}" in
        "deploy")
            check_prerequisites
            build_image
            deploy_services
            if wait_for_health; then
                show_status
                log_success "🎉 Deployment completed successfully!"
            else
                log_error "💥 Deployment failed health check"
                log_info "Use './deploy.sh rollback' to rollback"
                exit 1
            fi
            ;;
        "rollback")
            rollback
            ;;
        "status")
            show_status
            ;;
        "logs")
            docker-compose logs -f rtai
            ;;
        "stop")
            log_info "🛑 Stopping services..."
            docker-compose down
            log_success "✅ Services stopped"
            ;;
        "restart")
            log_info "🔄 Restarting services..."
            docker-compose restart
            wait_for_health && show_status
            ;;
        *)
            echo "Usage: $0 {deploy|rollback|status|logs|stop|restart}"
            echo ""
            echo "Commands:"
            echo "  deploy   - Deploy the RTAI trading system"
            echo "  rollback - Rollback to previous version"
            echo "  status   - Show service status and health"
            echo "  logs     - Show live logs"
            echo "  stop     - Stop all services"  
            echo "  restart  - Restart services"
            exit 1
            ;;
    esac
}

# Trap for cleanup on script interruption
trap 'log_warning "Deployment interrupted"; exit 130' INT

# Run main function
main "$@"
