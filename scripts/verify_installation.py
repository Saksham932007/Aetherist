#!/usr/bin/env python3
"""
Aetherist Installation Verification Script

This script validates that Aetherist has been installed correctly and 
all components are functioning as expected.

Usage:
    python scripts/verify_installation.py [--quick] [--verbose]
"""

import sys
import os
import subprocess
import importlib
import platform
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    END = '\033[0m'


class InstallationVerifier:
    """Comprehensive installation verification for Aetherist."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results = {}
        self.errors = []
        
    def print_status(self, message: str, status: str, details: str = ""):
        """Print formatted status message."""
        if status == "PASS":
            status_icon = f"{Colors.GREEN}✅{Colors.END}"
        elif status == "FAIL":
            status_icon = f"{Colors.RED}❌{Colors.END}"
        elif status == "WARN":
            status_icon = f"{Colors.YELLOW}⚠️{Colors.END}"
        else:
            status_icon = f"{Colors.BLUE}ℹ️{Colors.END}"
        
        print(f"{status_icon} {message}")
        if details and (self.verbose or status == "FAIL"):
            print(f"   {Colors.CYAN}{details}{Colors.END}")
    
    def check_python_version(self) -> bool:
        """Verify Python version compatibility."""
        version = sys.version_info
        version_str = f"{version.major}.{version.minor}.{version.micro}"
        
        if version.major == 3 and version.minor >= 8:
            self.print_status(
                f"Python Version: {version_str}",
                "PASS",
                "Compatible Python version"
            )
            return True
        else:
            self.print_status(
                f"Python Version: {version_str}",
                "FAIL", 
                "Python 3.8+ required"
            )
            self.errors.append("Incompatible Python version")
            return False
    
    def check_pytorch_installation(self) -> bool:
        """Verify PyTorch installation and CUDA availability."""
        try:
            import torch
            import torchvision
            
            pytorch_version = torch.__version__
            self.print_status(
                f"PyTorch Version: {pytorch_version}",
                "PASS",
                "PyTorch successfully imported"
            )
            
            # Check CUDA availability
            if torch.cuda.is_available():
                cuda_version = torch.version.cuda
                gpu_count = torch.cuda.device_count()
                gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
                
                self.print_status(
                    f"CUDA Support: Available (v{cuda_version})",
                    "PASS",
                    f"Found {gpu_count} GPU(s): {', '.join(gpu_names)}"
                )
            else:
                self.print_status(
                    "CUDA Support: Not Available",
                    "WARN",
                    "Will use CPU for computation (slower)"
                )
            
            # Check MPS availability (Apple Silicon)
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.print_status(
                    "Apple MPS Support: Available",
                    "PASS",
                    "Metal Performance Shaders available"
                )
            
            return True
            
        except ImportError as e:
            self.print_status(
                "PyTorch Installation",
                "FAIL",
                f"Import error: {str(e)}"
            )
            self.errors.append("PyTorch not installed or corrupted")
            return False
    
    def check_aetherist_installation(self) -> bool:
        """Verify Aetherist package installation."""
        try:
            import aetherist
            
            # Check version
            if hasattr(aetherist, '__version__'):
                version = aetherist.__version__
                self.print_status(
                    f"Aetherist Version: {version}",
                    "PASS",
                    f"Installed at: {aetherist.__file__}"
                )
            else:
                self.print_status(
                    "Aetherist Version: Unknown",
                    "WARN",
                    "Version information not available"
                )
            
            # Check core modules
            core_modules = [
                'aetherist.models',
                'aetherist.training',
                'aetherist.utils',
                'aetherist.data'
            ]
            
            for module_name in core_modules:
                try:
                    importlib.import_module(module_name)
                    self.print_status(
                        f"Module: {module_name}",
                        "PASS",
                        "Successfully imported"
                    )
                except ImportError as e:
                    self.print_status(
                        f"Module: {module_name}",
                        "FAIL",
                        f"Import error: {str(e)}"
                    )
                    self.errors.append(f"Missing module: {module_name}")
                    return False
            
            return True
            
        except ImportError as e:
            self.print_status(
                "Aetherist Installation",
                "FAIL",
                f"Package not found: {str(e)}"
            )
            self.errors.append("Aetherist package not installed")
            return False
    
    def check_dependencies(self) -> bool:
        """Check all required dependencies."""
        dependencies = {
            'numpy': 'Numerical computing',
            'PIL': 'Image processing', 
            'matplotlib': 'Plotting and visualization',
            'tqdm': 'Progress bars',
            'yaml': 'Configuration files',
            'requests': 'HTTP requests',
            'click': 'Command line interface'
        }
        
        optional_dependencies = {
            'cv2': 'OpenCV for computer vision',
            'gradio': 'Web interface',
            'fastapi': 'REST API',
            'uvicorn': 'ASGI server',
            'redis': 'Caching',
            'tensorboard': 'Training visualization'
        }
        
        all_good = True
        
        # Check required dependencies
        for package, description in dependencies.items():
            try:
                importlib.import_module(package)
                self.print_status(
                    f"Dependency: {package}",
                    "PASS",
                    description
                )
            except ImportError:
                self.print_status(
                    f"Dependency: {package}",
                    "FAIL",
                    f"Required for: {description}"
                )
                self.errors.append(f"Missing dependency: {package}")
                all_good = False
        
        # Check optional dependencies
        for package, description in optional_dependencies.items():
            try:
                importlib.import_module(package)
                self.print_status(
                    f"Optional: {package}",
                    "PASS",
                    description
                )
            except ImportError:
                self.print_status(
                    f"Optional: {package}",
                    "WARN",
                    f"Optional for: {description}"
                )
        
        return all_good
    
    def check_model_files(self) -> bool:
        """Check for model files and weights."""
        model_paths = [
            "models/",
            "models/aetherist_v1.pth",
            "models/discriminator.pth",
            "models/attribute_vectors/"
        ]
        
        found_any = False
        
        for path in model_paths:
            if os.path.exists(path):
                if os.path.isfile(path):
                    size = os.path.getsize(path) / (1024**3)  # GB
                    self.print_status(
                        f"Model File: {path}",
                        "PASS",
                        f"Size: {size:.2f} GB"
                    )
                    found_any = True
                else:
                    self.print_status(
                        f"Model Directory: {path}",
                        "PASS",
                        "Directory exists"
                    )
                    found_any = True
            else:
                self.print_status(
                    f"Model Path: {path}",
                    "WARN",
                    "File/directory not found"
                )
        
        if not found_any:
            self.print_status(
                "Model Files",
                "WARN",
                "No model files found. You may need to download them."
            )
        
        return True  # Not critical for basic functionality
    
    def check_configuration_files(self) -> bool:
        """Check for configuration files."""
        config_files = [
            "configs/model_config.yaml",
            "configs/train_config.yaml", 
            "configs/api_config.yaml"
        ]
        
        found_any = False
        
        for config_file in config_files:
            if os.path.exists(config_file):
                self.print_status(
                    f"Config: {config_file}",
                    "PASS",
                    "Configuration file found"
                )
                found_any = True
            else:
                self.print_status(
                    f"Config: {config_file}",
                    "WARN",
                    "Configuration file not found"
                )
        
        if not found_any:
            self.print_status(
                "Configuration Files",
                "WARN",
                "No configuration files found. Using defaults."
            )
        
        return True
    
    def test_model_initialization(self) -> bool:
        """Test basic model initialization."""
        try:
            from aetherist.models.generator import AetheristGenerator
            from aetherist.config import AetheristConfig
            
            # Create minimal config for testing
            config = AetheristConfig(
                latent_dim=512,
                triplane_dim=256,
                triplane_res=32,  # Small for quick test
                resolution=256     # Small for quick test
            )
            
            # Initialize generator
            generator = AetheristGenerator(config)
            
            self.print_status(
                "Model Initialization",
                "PASS",
                "Generator created successfully"
            )
            
            # Test forward pass with dummy data
            import torch
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            generator = generator.to(device)
            
            # Create dummy inputs
            latent_code = torch.randn(1, 512, device=device)
            camera_params = torch.randn(1, 25, device=device)
            
            # Test forward pass
            with torch.no_grad():
                output = generator(latent_code, camera_params)
            
            self.print_status(
                "Model Forward Pass",
                "PASS",
                f"Output shape: {output.shape}"
            )
            
            return True
            
        except Exception as e:
            self.print_status(
                "Model Testing",
                "FAIL",
                f"Error: {str(e)}"
            )
            self.errors.append(f"Model test failed: {str(e)}")
            return False
    
    def test_api_availability(self) -> bool:
        """Test if API components are available."""
        try:
            from aetherist.api.main import app
            
            self.print_status(
                "API Components",
                "PASS", 
                "FastAPI application available"
            )
            
            # Test if server can be imported
            try:
                import uvicorn
                self.print_status(
                    "ASGI Server",
                    "PASS",
                    "Uvicorn server available"
                )
            except ImportError:
                self.print_status(
                    "ASGI Server",
                    "WARN",
                    "Uvicorn not available (API won't work)"
                )
            
            return True
            
        except Exception as e:
            self.print_status(
                "API Testing",
                "WARN",
                f"API components not available: {str(e)}"
            )
            return True  # Not critical for core functionality
    
    def check_system_resources(self) -> bool:
        """Check system resources and capabilities."""
        import psutil
        
        # Memory check
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        
        if memory_gb >= 16:
            status = "PASS"
            detail = "Sufficient memory for training"
        elif memory_gb >= 8:
            status = "WARN" 
            detail = "Limited memory, reduce batch sizes"
        else:
            status = "FAIL"
            detail = "Insufficient memory for training"
            
        self.print_status(
            f"System Memory: {memory_gb:.1f} GB",
            status,
            detail
        )
        
        # Disk space check
        disk = psutil.disk_usage('/')
        free_gb = disk.free / (1024**3)
        
        if free_gb >= 50:
            status = "PASS"
            detail = "Sufficient disk space"
        elif free_gb >= 20:
            status = "WARN"
            detail = "Limited disk space"
        else:
            status = "FAIL"
            detail = "Insufficient disk space"
            
        self.print_status(
            f"Free Disk Space: {free_gb:.1f} GB", 
            status,
            detail
        )
        
        # CPU check
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()
        
        if cpu_freq:
            freq_ghz = cpu_freq.current / 1000
            self.print_status(
                f"CPU: {cpu_count} cores @ {freq_ghz:.1f} GHz",
                "PASS" if cpu_count >= 4 else "WARN",
                "Good for training" if cpu_count >= 8 else "Consider more cores"
            )
        else:
            self.print_status(
                f"CPU: {cpu_count} cores",
                "PASS" if cpu_count >= 4 else "WARN",
                "CPU frequency not available"
            )
        
        return True
    
    def run_verification(self, quick: bool = False) -> bool:
        """Run complete installation verification."""
        print(f"{Colors.BOLD}{Colors.BLUE}🔍 Aetherist Installation Verification{Colors.END}")
        print("=" * 50)
        
        # System information
        print(f"\n{Colors.BOLD}System Information:{Colors.END}")
        print(f"Platform: {platform.platform()}")
        print(f"Python: {sys.version}")
        print(f"Architecture: {platform.machine()}")
        
        # Run verification checks
        checks = [
            ("Python Version", self.check_python_version),
            ("PyTorch Installation", self.check_pytorch_installation),
            ("Aetherist Package", self.check_aetherist_installation),
            ("Dependencies", self.check_dependencies),
            ("Configuration Files", self.check_configuration_files),
            ("System Resources", self.check_system_resources),
        ]
        
        if not quick:
            checks.extend([
                ("Model Files", self.check_model_files),
                ("Model Initialization", self.test_model_initialization),
                ("API Components", self.test_api_availability),
            ])
        
        print(f"\n{Colors.BOLD}Running Verification Checks:{Colors.END}")
        
        all_passed = True
        for check_name, check_func in checks:
            print(f"\n{Colors.PURPLE}📋 {check_name}:{Colors.END}")
            try:
                result = check_func()
                all_passed = all_passed and result
            except Exception as e:
                self.print_status(
                    check_name,
                    "FAIL",
                    f"Unexpected error: {str(e)}"
                )
                self.errors.append(f"{check_name}: {str(e)}")
                all_passed = False
        
        # Print summary
        print(f"\n{Colors.BOLD}Verification Summary:{Colors.END}")
        print("=" * 50)
        
        if all_passed and not self.errors:
            print(f"{Colors.GREEN}{Colors.BOLD}✅ All checks passed! Aetherist is ready to use.{Colors.END}")
            print(f"\n{Colors.CYAN}Next steps:{Colors.END}")
            print("1. Download model weights if needed")
            print("2. Review configuration files")
            print("3. Try the getting started guide")
            print("4. Run: python -c \"from aetherist import quick_generate; quick_generate()\"")
            
        elif not self.errors:
            print(f"{Colors.YELLOW}{Colors.BOLD}⚠️  Installation complete with warnings.{Colors.END}")
            print("Some optional components are missing but core functionality should work.")
            
        else:
            print(f"{Colors.RED}{Colors.BOLD}❌ Installation verification failed.{Colors.END}")
            print(f"\n{Colors.RED}Errors found:{Colors.END}")
            for error in self.errors:
                print(f"  • {error}")
            
            print(f"\n{Colors.CYAN}Suggested fixes:{Colors.END}")
            print("1. Check the installation guide: docs/installation.md") 
            print("2. Ensure all dependencies are installed: pip install -r requirements.txt")
            print("3. Verify Python version compatibility (3.8+)")
            print("4. Check GPU drivers if using CUDA")
        
        return all_passed and not self.errors


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(
        description="Verify Aetherist installation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/verify_installation.py              # Full verification
  python scripts/verify_installation.py --quick      # Quick checks only
  python scripts/verify_installation.py --verbose    # Detailed output
        """
    )
    
    parser.add_argument(
        "--quick", 
        action="store_true",
        help="Run quick verification (skip model tests)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true", 
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Run verification
    verifier = InstallationVerifier(verbose=args.verbose)
    success = verifier.run_verification(quick=args.quick)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()