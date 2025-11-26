#!/bin/bash

# Aetherist Deployment Script
# Supports Docker Compose and Kubernetes deployments

set -euo pipefail

# Configuration
DEPLOYMENT_TYPE="docker-compose"  # docker-compose, kubernetes, local
ENVIRONMENT="production"  # development, staging, production
IMAGE_TAG="latest"
REGISTRY=""
NAMESPACE="aetherist"

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

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -t|--type)
                DEPLOYMENT_TYPE="$2"
                shift 2
                ;;
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            --tag)
                IMAGE_TAG="$2"
                shift 2
                ;;
            --registry)
                REGISTRY="$2"
                shift 2
                ;;
            --namespace)
                NAMESPACE="$2"
                shift 2
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown argument: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

show_help() {
    cat << EOF
Aetherist Deployment Script

Usage: $0 [OPTIONS]

Options:
    -t, --type TYPE         Deployment type (docker-compose, kubernetes, local)
    -e, --environment ENV   Environment (development, staging, production)
    --tag TAG              Docker image tag (default: latest)
    --registry REGISTRY    Docker registry prefix
    --namespace NAMESPACE  Kubernetes namespace (default: aetherist)
    -h, --help             Show this help message

Examples:
    $0 --type docker-compose --environment production
    $0 --type kubernetes --environment staging --tag v1.2.3
    $0 --type local --environment development
EOF
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    case $DEPLOYMENT_TYPE in
        docker-compose)
            if ! command -v docker &> /dev/null; then
                log_error "Docker is not installed"
                exit 1
            fi
            if ! command -v docker-compose &> /dev/null; then
                log_error "Docker Compose is not installed"
                exit 1
            fi
            ;;
        kubernetes)
            if ! command -v kubectl &> /dev/null; then
                log_error "kubectl is not installed"
                exit 1
            fi
            if ! kubectl cluster-info &> /dev/null; then
                log_error "Cannot connect to Kubernetes cluster"
                exit 1
            fi
            ;;
        local)
            if ! command -v python &> /dev/null; then
                log_error "Python is not installed"
                exit 1
            fi
            ;;
    esac
    
    log_success "Prerequisites check passed"
}

# Build Docker image
build_image() {
    log_info "Building Docker image..."
    
    local image_name="aetherist:${IMAGE_TAG}"
    if [[ -n "$REGISTRY" ]]; then
        image_name="${REGISTRY}/aetherist:${IMAGE_TAG}"
    fi
    
    docker build -t "$image_name" .
    
    log_success "Docker image built: $image_name"
    
    # Push to registry if specified
    if [[ -n "$REGISTRY" ]]; then
        log_info "Pushing image to registry..."
        docker push "$image_name"
        log_success "Image pushed to registry"
    fi
}

# Deploy with Docker Compose
deploy_docker_compose() {
    log_info "Deploying with Docker Compose..."
    
    # Create required directories
    mkdir -p data models outputs logs
    
    # Set environment variables
    export ENVIRONMENT
    export IMAGE_TAG
    
    # Deploy
    docker-compose down --remove-orphans
    docker-compose up -d
    
    # Wait for services to be ready
    log_info "Waiting for services to be ready..."
    sleep 30
    
    # Health check
    if curl -f http://localhost:8000/health &> /dev/null; then
        log_success "Aetherist API is healthy"
    else
        log_error "Aetherist API health check failed"
        docker-compose logs aetherist-api
        exit 1
    fi
    
    log_success "Docker Compose deployment completed"
}

# Deploy to Kubernetes
deploy_kubernetes() {
    log_info "Deploying to Kubernetes..."
    
    # Create namespace if it doesn't exist
    kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
    
    # Update image tag in deployment
    local image_name="aetherist:${IMAGE_TAG}"
    if [[ -n "$REGISTRY" ]]; then
        image_name="${REGISTRY}/aetherist:${IMAGE_TAG}"
    fi
    
    # Apply Kubernetes manifests
    kubectl apply -f k8s/ -n "$NAMESPACE"
    
    # Update deployment with new image
    kubectl set image deployment/aetherist-api aetherist-api="$image_name" -n "$NAMESPACE"
    kubectl set image deployment/aetherist-worker aetherist-worker="$image_name" -n "$NAMESPACE"
    
    # Wait for rollout to complete
    log_info "Waiting for deployment rollout..."
    kubectl rollout status deployment/aetherist-api -n "$NAMESPACE" --timeout=300s
    kubectl rollout status deployment/aetherist-worker -n "$NAMESPACE" --timeout=300s
    
    # Health check
    log_info "Performing health check..."
    local service_url
    service_url=$(kubectl get service aetherist-api-service -n "$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    
    if [[ -n "$service_url" ]]; then
        if curl -f "http://$service_url/health" &> /dev/null; then
            log_success "Aetherist API is healthy"
        else
            log_warning "External health check failed, checking pod status..."
        fi
    fi
    
    # Show deployment status
    kubectl get pods -n "$NAMESPACE" -l app=aetherist-api
    kubectl get pods -n "$NAMESPACE" -l app=aetherist-worker
    
    log_success "Kubernetes deployment completed"
}

# Deploy locally
deploy_local() {
    log_info "Setting up local development environment..."
    
    # Create virtual environment if it doesn't exist
    if [[ ! -d "venv" ]]; then
        log_info "Creating Python virtual environment..."
        python -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Install dependencies
    log_info "Installing dependencies..."
    pip install -r requirements.txt
    pip install -e .
    
    # Create required directories
    mkdir -p data models outputs logs
    
    # Set environment variables
    export ENVIRONMENT
    
    # Start the application
    log_info "Starting Aetherist locally..."
    
    # Start background services
    if command -v redis-server &> /dev/null; then
        redis-server --daemonize yes
        log_info "Redis started"
    else
        log_warning "Redis not found, some features may not work"
    fi
    
    # Start API server
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload &
    API_PID=$!
    
    # Wait for API to start
    sleep 5
    
    # Health check
    if curl -f http://localhost:8000/health &> /dev/null; then
        log_success "Aetherist API is running at http://localhost:8000"
        log_info "API Documentation available at http://localhost:8000/docs"
        log_info "API PID: $API_PID"
        
        # Save PID for easy stopping
        echo $API_PID > aetherist.pid
        
    else
        log_error "Failed to start Aetherist API"
        kill $API_PID 2>/dev/null || true
        exit 1
    fi
    
    log_success "Local deployment completed"
}

# Cleanup function
cleanup() {
    log_info "Performing cleanup..."
    
    case $DEPLOYMENT_TYPE in
        docker-compose)
            docker-compose down --remove-orphans
            ;;
        kubernetes)
            kubectl delete -f k8s/ -n "$NAMESPACE" --ignore-not-found=true
            ;;
        local)
            if [[ -f "aetherist.pid" ]]; then
                local pid=$(cat aetherist.pid)
                kill "$pid" 2>/dev/null || true
                rm aetherist.pid
            fi
            ;;
    esac
    
    log_success "Cleanup completed"
}

# Rollback function
rollback() {
    log_info "Rolling back deployment..."
    
    case $DEPLOYMENT_TYPE in
        kubernetes)
            kubectl rollout undo deployment/aetherist-api -n "$NAMESPACE"
            kubectl rollout undo deployment/aetherist-worker -n "$NAMESPACE"
            kubectl rollout status deployment/aetherist-api -n "$NAMESPACE"
            kubectl rollout status deployment/aetherist-worker -n "$NAMESPACE"
            ;;
        *)
            log_warning "Rollback not supported for $DEPLOYMENT_TYPE"
            ;;
    esac
    
    log_success "Rollback completed"
}

# Status check
check_status() {
    log_info "Checking deployment status..."
    
    case $DEPLOYMENT_TYPE in
        docker-compose)
            docker-compose ps
            ;;
        kubernetes)
            kubectl get pods -n "$NAMESPACE"
            kubectl get services -n "$NAMESPACE"
            ;;
        local)
            if [[ -f "aetherist.pid" ]]; then
                local pid=$(cat aetherist.pid)
                if ps -p "$pid" > /dev/null; then
                    log_success "Aetherist API is running (PID: $pid)"
                else
                    log_error "Aetherist API is not running"
                fi
            else
                log_error "Aetherist API is not running"
            fi
            ;;
    esac
}

# Main deployment function
main() {
    log_info "Starting Aetherist deployment"
    log_info "Deployment type: $DEPLOYMENT_TYPE"
    log_info "Environment: $ENVIRONMENT"
    log_info "Image tag: $IMAGE_TAG"
    
    check_prerequisites
    
    case $DEPLOYMENT_TYPE in
        docker-compose)
            build_image
            deploy_docker_compose
            ;;
        kubernetes)
            build_image
            deploy_kubernetes
            ;;
        local)
            deploy_local
            ;;
        *)
            log_error "Unknown deployment type: $DEPLOYMENT_TYPE"
            exit 1
            ;;
    esac
    
    log_success "Deployment completed successfully!"
}

# Handle special commands
if [[ $# -gt 0 ]]; then
    case $1 in
        status)
            check_status
            exit 0
            ;;
        cleanup)
            cleanup
            exit 0
            ;;
        rollback)
            shift
            parse_args "$@"
            rollback
            exit 0
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
    esac
fi

# Parse arguments and run main deployment
parse_args "$@"
main
