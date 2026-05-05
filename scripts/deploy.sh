#!/bin/bash
# ============================================================
#  MNIST Deployment Script
#  Arch Technologies Internship Project
# ============================================================
#  One-command deployment with multiple options
# ============================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="mnist-recognition"
DOCKER_IMAGE="$PROJECT_NAME:latest"
FASTAPI_PORT=8000
FLASK_PORT=5000

# Helper functions
print_header() {
    echo -e "${BLUE}============================================================${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}============================================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# Check prerequisites
check_prerequisites() {
    print_header "Checking Prerequisites"
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker not installed. Please install Docker first."
        exit 1
    fi
    print_success "Docker found"
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose not installed. Please install Docker Compose first."
        exit 1
    fi
    print_success "Docker Compose found"
    
    # Check if model exists
    if [ ! -f "models/cnn_tuned.keras" ] && [ ! -f "models/cnn_mnist.keras" ]; then
        print_warning "No trained models found. Please run training first:"
        echo "  python src/cnn_full_dataset_optimized.py"
        echo "  python src/hyperparameter_tuning_keras_tuner.py"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        print_success "Trained models found"
    fi
}

# Build Docker images
build_images() {
    print_header "Building Docker Images"
    
    echo "Building main application image..."
    docker build -t $DOCKER_IMAGE .
    print_success "Docker image built successfully"
}

# Deploy with Docker Compose
deploy_compose() {
    print_header "Deploying with Docker Compose"
    
    # Stop existing services
    echo "Stopping existing services..."
    docker-compose down 2>/dev/null || true
    
    # Start services
    echo "Starting services..."
    docker-compose up -d
    
    # Wait for services to be ready
    echo "Waiting for services to be ready..."
    sleep 10
    
    # Check service health
    if curl -f http://localhost:$FLASK_PORT/health &> /dev/null; then
        print_success "Flask app is healthy"
    else
        print_error "Flask app health check failed"
    fi
    
    if curl -f http://localhost:$FASTAPI_PORT/health &> /dev/null; then
        print_success "FastAPI app is healthy"
    else
        print_warning "FastAPI app not available (not deployed by default)"
    fi
}

# Deploy Flask only
deploy_flask() {
    print_header "Deploying Flask Application"
    
    # Stop existing container
    echo "Stopping existing Flask container..."
    docker stop mnist-flask 2>/dev/null || true
    docker rm mnist-flask 2>/dev/null || true
    
    # Run new container
    echo "Starting Flask container..."
    docker run -d \
        --name mnist-flask \
        -p $FLASK_PORT:5000 \
        -v $(pwd)/logs:/app/logs \
        -v $(pwd)/outputs:/app/outputs \
        $DOCKER_IMAGE
    
    # Wait for startup
    sleep 5
    
    # Health check
    if curl -f http://localhost:$FLASK_PORT/health &> /dev/null; then
        print_success "Flask app deployed successfully"
    else
        print_error "Flask app deployment failed"
        docker logs mnist-flask
        exit 1
    fi
}

# Deploy FastAPI
deploy_fastapi() {
    print_header "Deploying FastAPI Application"
    
    # Build FastAPI image
    echo "Building FastAPI image..."
    cd fastapi_app
    docker build -t $PROJECT_NAME-fastapi .
    cd ..
    
    # Stop existing container
    echo "Stopping existing FastAPI container..."
    docker stop mnist-fastapi 2>/dev/null || true
    docker rm mnist-fastapi 2>/dev/null || true
    
    # Run new container
    echo "Starting FastAPI container..."
    docker run -d \
        --name mnist-fastapi \
        -p $FASTAPI_PORT:8000 \
        -v $(pwd)/models:/app/models \
        $PROJECT_NAME-fastapi
    
    # Wait for startup
    sleep 5
    
    # Health check
    if curl -f http://localhost:$FASTAPI_PORT/health &> /dev/null; then
        print_success "FastAPI app deployed successfully"
    else
        print_error "FastAPI app deployment failed"
        docker logs mnist-fastapi
        exit 1
    fi
}

# Convert models for deployment
convert_models() {
    print_header "Converting Models for Deployment"
    
    # TensorFlow Lite
    echo "Converting to TensorFlow Lite..."
    python src/convert_to_tflite.py
    print_success "TensorFlow Lite models created"
    
    # ONNX
    echo "Converting to ONNX..."
    python src/convert_to_onnx.py
    print_success "ONNX models created"
}

# Run tests
run_tests() {
    print_header "Running Deployment Tests"
    
    echo "Testing Flask endpoint..."
    response=$(curl -s http://localhost:$FLASK_PORT/health)
    if [[ $response == *"ok"* ]]; then
        print_success "Flask health check passed"
    else
        print_error "Flask health check failed"
    fi
    
    echo "Testing FastAPI endpoint..."
    response=$(curl -s http://localhost:$FASTAPI_PORT/health)
    if [[ $response == *"ok"* ]]; then
        print_success "FastAPI health check passed"
    else
        print_warning "FastAPI not available"
    fi
}

# Show deployment status
show_status() {
    print_header "Deployment Status"
    
    echo "Running containers:"
    docker ps --filter "name=mnist" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    
    echo -e "\nService URLs:"
    echo -e "  Flask App:     ${GREEN}http://localhost:$FLASK_PORT${NC}"
    echo -e "  FastAPI App:   ${GREEN}http://localhost:$FASTAPI_PORT${NC}"
    echo -e "  Grafana:       ${GREEN}http://localhost:3000${NC} (admin/admin)"
    echo -e "  Prometheus:    ${GREEN}http://localhost:9090${NC}"
    
    echo -e "\nAPI Documentation:"
    echo -e "  Flask:         ${GREEN}http://localhost:$FLASK_PORT${NC}"
    echo -e "  FastAPI Docs:  ${GREEN}http://localhost:$FASTAPI_PORT/docs${NC}"
    echo -e "  FastAPI ReDoc: ${GREEN}http://localhost:$FASTAPI_PORT/redoc${NC}"
}

# Cleanup deployment
cleanup() {
    print_header "Cleaning Up Deployment"
    
    echo "Stopping and removing containers..."
    docker-compose down 2>/dev/null || true
    docker stop mnist-flask mnist-fastapi 2>/dev/null || true
    docker rm mnist-flask mnist-fastapi 2>/dev/null || true
    
    echo "Removing Docker images..."
    docker rmi $DOCKER_IMAGE 2>/dev/null || true
    docker rmi $PROJECT_NAME-fastapi 2>/dev/null || true
    
    print_success "Cleanup completed"
}

# Show usage
show_usage() {
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  compose     Deploy full stack with Docker Compose"
    echo "  flask       Deploy Flask app only"
    echo "  fastapi     Deploy FastAPI app only"
    echo "  convert     Convert models (TFLite, ONNX)"
    echo "  test        Run deployment tests"
    echo "  status      Show deployment status"
    echo "  cleanup     Clean up deployment"
    echo "  all         Build, convert, and deploy everything"
    echo ""
    echo "Examples:"
    echo "  $0 compose      # Deploy full production stack"
    echo "  $0 flask        # Deploy Flask only"
    echo "  $0 fastapi      # Deploy FastAPI only"
    echo "  $0 convert      # Convert models for mobile/production"
    echo "  $0 all          # Complete deployment pipeline"
}

# Main script logic
case "${1:-}" in
    "compose")
        check_prerequisites
        build_images
        deploy_compose
        show_status
        ;;
    "flask")
        check_prerequisites
        build_images
        deploy_flask
        show_status
        ;;
    "fastapi")
        check_prerequisites
        deploy_fastapi
        show_status
        ;;
    "convert")
        convert_models
        ;;
    "test")
        run_tests
        ;;
    "status")
        show_status
        ;;
    "cleanup")
        cleanup
        ;;
    "all")
        check_prerequisites
        build_images
        convert_models
        deploy_compose
        run_tests
        show_status
        ;;
    "help"|"--help"|"-h")
        show_usage
        ;;
    "")
        print_error "No command specified"
        show_usage
        exit 1
        ;;
    *)
        print_error "Unknown command: $1"
        show_usage
        exit 1
        ;;
esac

echo -e "\n${GREEN}Deployment completed successfully!${NC}"
