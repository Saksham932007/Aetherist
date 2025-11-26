#!/usr/bin/env bash
# Aetherist Quick Setup Script
# One-command setup for development and production environments

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color

# Default values
ENVIRONMENT="development"
SKIP_MODELS=false
SKIP_TESTS=false
QUICK_MODE=false
GPU_SUPPORT="auto"

# Print colored output
print_header() {
    echo -e "${PURPLE}================================${NC}"
    echo -e "${WHITE}🚀 Aetherist Quick Setup${NC}"
    echo -e "${PURPLE}================================${NC}"
}

print_step() {
    echo -e "${BLUE}📋 $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${CYAN}ℹ️ $1${NC}"
}

# Show usage
show_usage() {
    cat << EOF
Aetherist Quick Setup Script

USAGE:
    ./setup.sh [OPTIONS]

OPTIONS:
    -e, --environment ENV     Environment: development|production|docker (default: development)
    -g, --gpu SUPPORT        GPU support: auto|cuda|cpu|mps (default: auto)
    -q, --quick              Quick mode: skip optional components
    -s, --skip-models        Skip model downloads
    -t, --skip-tests         Skip running tests
    -h, --help               Show this help message

EXAMPLES:
    ./setup.sh                           # Development setup with auto-detection
    ./setup.sh -e production -g cuda     # Production setup with CUDA
    ./setup.sh -q -s                     # Quick setup without models
    ./setup.sh -e docker                 # Docker-based setup

ENVIRONMENTS:
    development    Full development environment with tools
    production     Optimized production deployment
    docker         Docker-based containerized setup
EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -g|--gpu)
                GPU_SUPPORT="$2"
                shift 2
                ;;
            -q|--quick)
                QUICK_MODE=true
                shift
                ;;
            -s|--skip-models)
                SKIP_MODELS=true
                shift
                ;;
            -t|--skip-tests)
                SKIP_TESTS=true
                shift
                ;;
            -h|--help)
                show_usage
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
}

# Detect system information
detect_system() {
    print_step "Detecting system information..."
    
    # Operating system
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
        OS="windows"
    else
        OS="unknown"
    fi
    
    # Python version
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
    else
        print_error "Python not found. Please install Python 3.8+"
        exit 1
    fi
    
    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2)
    
    # Package managers
    PACKAGE_MANAGER="pip"
    if command -v conda &> /dev/null; then
        PACKAGE_MANAGER="conda"
    elif command -v poetry &> /dev/null; then
        PACKAGE_MANAGER="poetry"
    fi
    
    # GPU detection
    if [[ "$GPU_SUPPORT" == "auto" ]]; then
        if command -v nvidia-smi &> /dev/null; then
            GPU_SUPPORT="cuda"
        elif [[ "$OS" == "macos" ]]; then
            GPU_SUPPORT="mps"
        else
            GPU_SUPPORT="cpu"
        fi
    fi
    
    print_info "OS: $OS"
    print_info "Python: $PYTHON_VERSION ($PYTHON_CMD)"
    print_info "Package Manager: $PACKAGE_MANAGER"
    print_info "GPU Support: $GPU_SUPPORT"
}

# Check prerequisites
check_prerequisites() {
    print_step "Checking prerequisites..."
    
    # Check Python version
    if $PYTHON_CMD -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)" 2>/dev/null; then
        print_success "Python version is compatible"
    else
        print_error "Python 3.8+ required. Found: $PYTHON_VERSION"
        exit 1
    fi
    
    # Check git
    if command -v git &> /dev/null; then
        print_success "Git is available"
    else
        print_error "Git is required but not installed"
        exit 1
    fi
    
    # Check disk space (need at least 10GB)
    if command -v df &> /dev/null; then
        AVAILABLE_SPACE=$(df . | tail -1 | awk '{print $4}')
        if [[ $AVAILABLE_SPACE -gt 10485760 ]]; then  # 10GB in KB
            print_success "Sufficient disk space available"
        else
            print_warning "Low disk space. At least 10GB recommended"
        fi
    fi
    
    # Check memory (recommend 8GB)
    if command -v free &> /dev/null; then
        TOTAL_MEM=$(free -g | awk 'NR==2{print $2}')
        if [[ $TOTAL_MEM -ge 8 ]]; then
            print_success "Sufficient memory available (${TOTAL_MEM}GB)"
        else
            print_warning "Low memory. 8GB+ recommended. Found: ${TOTAL_MEM}GB"
        fi
    fi
}

# Setup Python environment
setup_python_environment() {
    print_step "Setting up Python environment..."
    
    case $PACKAGE_MANAGER in
        "conda")
            if conda info --envs | grep -q "aetherist"; then
                print_info "Conda environment 'aetherist' already exists"
                conda activate aetherist 2>/dev/null || true
            else
                print_info "Creating conda environment..."
                conda create -n aetherist python=3.9 -y
                conda activate aetherist
            fi
            ;;
        "poetry")
            if [[ -f "pyproject.toml" ]]; then
                print_info "Using existing Poetry configuration"
            else
                print_info "Initializing Poetry project..."
                poetry init --no-interaction --name aetherist --version 1.0.0
            fi
            poetry install
            ;;
        "pip")
            if [[ ! -d "venv" ]]; then
                print_info "Creating virtual environment..."
                $PYTHON_CMD -m venv venv
            fi
            
            # Activate virtual environment
            if [[ "$OS" == "windows" ]]; then
                source venv/Scripts/activate
            else
                source venv/bin/activate
            fi
            
            # Upgrade pip
            pip install --upgrade pip setuptools wheel
            ;;
    esac
    
    print_success "Python environment ready"
}

# Install dependencies
install_dependencies() {
    print_step "Installing dependencies..."
    
    # Install PyTorch with appropriate support
    case $GPU_SUPPORT in
        "cuda")
            print_info "Installing PyTorch with CUDA support..."
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
            ;;
        "mps")
            print_info "Installing PyTorch with MPS support..."
            pip install torch torchvision torchaudio
            ;;
        "cpu")
            print_info "Installing PyTorch (CPU only)..."
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
            ;;
    esac
    
    # Install main requirements
    if [[ -f "requirements.txt" ]]; then
        pip install -r requirements.txt
    fi
    
    # Install development requirements for dev environment
    if [[ "$ENVIRONMENT" == "development" ]] && [[ -f "requirements-dev.txt" ]]; then
        pip install -r requirements-dev.txt
    fi
    
    # Install package in editable mode
    pip install -e .
    
    print_success "Dependencies installed"
}

# Download models
download_models() {
    if [[ "$SKIP_MODELS" == "true" ]]; then
        print_info "Skipping model downloads"
        return
    fi
    
    print_step "Downloading model weights..."
    
    # Create models directory
    mkdir -p models
    
    # Use the model download script if available
    if [[ -f "scripts/download_models.py" ]]; then
        $PYTHON_CMD scripts/download_models.py --quick
    else
        print_warning "Model download script not found. Models will be downloaded on first use."
    fi
    
    print_success "Model setup complete"
}

# Setup development tools
setup_dev_tools() {
    if [[ "$ENVIRONMENT" != "development" ]] || [[ "$QUICK_MODE" == "true" ]]; then
        return
    fi
    
    print_step "Setting up development tools..."
    
    # Pre-commit hooks
    if command -v pre-commit &> /dev/null; then
        if [[ -f ".pre-commit-config.yaml" ]]; then
            pre-commit install
            print_success "Pre-commit hooks installed"
        fi
    fi
    
    # VS Code settings
    if [[ -d ".vscode" ]]; then
        print_success "VS Code configuration found"
    fi
    
    print_success "Development tools ready"
}

# Run verification
run_verification() {
    if [[ "$SKIP_TESTS" == "true" ]]; then
        print_info "Skipping verification tests"
        return
    fi
    
    print_step "Running installation verification..."
    
    if [[ -f "scripts/verify_installation.py" ]]; then
        if [[ "$QUICK_MODE" == "true" ]]; then
            $PYTHON_CMD scripts/verify_installation.py --quick
        else
            $PYTHON_CMD scripts/verify_installation.py
        fi
    else
        # Basic verification
        $PYTHON_CMD -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
try:
    import aetherist
    print('Aetherist package imported successfully')
except ImportError as e:
    print(f'Warning: Could not import aetherist: {e}')
"
    fi
    
    print_success "Verification complete"
}

# Docker setup
setup_docker() {
    if [[ "$ENVIRONMENT" != "docker" ]]; then
        return
    fi
    
    print_step "Setting up Docker environment..."
    
    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        print_error "Docker not found. Please install Docker first."
        exit 1
    fi
    
    # Check if docker-compose is available
    if command -v docker-compose &> /dev/null; then
        COMPOSE_CMD="docker-compose"
    elif docker compose version &> /dev/null; then
        COMPOSE_CMD="docker compose"
    else
        print_error "Docker Compose not found. Please install Docker Compose."
        exit 1
    fi
    
    # Build and start services
    print_info "Building Docker images..."
    $COMPOSE_CMD build
    
    print_info "Starting services..."
    $COMPOSE_CMD up -d
    
    print_success "Docker environment ready"
    print_info "API available at: http://localhost:8000"
    print_info "Web interface at: http://localhost:7860"
}

# Print completion message
print_completion() {
    echo ""
    echo -e "${GREEN}🎉 Setup completed successfully!${NC}"
    echo -e "${PURPLE}================================${NC}"
    
    case $ENVIRONMENT in
        "development")
            echo -e "${CYAN}Next steps for development:${NC}"
            echo "  1. Activate environment: source venv/bin/activate (or conda activate aetherist)"
            echo "  2. Run examples: python examples/basic_usage.py"
            echo "  3. Start API: python -m uvicorn aetherist.api.main:app --reload"
            echo "  4. Launch web UI: python -m aetherist.web.gradio_app"
            ;;
        "production")
            echo -e "${CYAN}Production deployment ready:${NC}"
            echo "  1. Start API: python -m uvicorn aetherist.api.main:app --workers 4"
            echo "  2. Check health: curl http://localhost:8000/health"
            echo "  3. View docs: http://localhost:8000/docs"
            ;;
        "docker")
            echo -e "${CYAN}Docker services running:${NC}"
            echo "  • API: http://localhost:8000"
            echo "  • Web UI: http://localhost:7860"
            echo "  • Docs: http://localhost:8000/docs"
            echo "  • Manage: docker-compose ps"
            ;;
    esac
    
    echo ""
    echo -e "${YELLOW}📚 Resources:${NC}"
    echo "  • Documentation: docs/"
    echo "  • Examples: examples/"
    echo "  • FAQ: FAQ.md"
    echo "  • Issues: https://github.com/aetherist/aetherist/issues"
}

# Main execution
main() {
    print_header
    
    # Parse arguments
    parse_args "$@"
    
    # Setup process
    detect_system
    check_prerequisites
    
    if [[ "$ENVIRONMENT" == "docker" ]]; then
        setup_docker
    else
        setup_python_environment
        install_dependencies
        download_models
        setup_dev_tools
        run_verification
    fi
    
    print_completion
}

# Execute main function with all arguments
main "$@"