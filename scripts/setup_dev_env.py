#!/usr/bin/env python3
"""
Aetherist Development Environment Setup Script

This script helps set up a complete development environment for Aetherist,
including all dependencies, pre-commit hooks, and development tools.

Usage:
    python scripts/setup_dev_env.py [--skip-cuda] [--skip-optional]
"""

import os
import sys
import subprocess
import platform
import argparse
from pathlib import Path
from typing import List, Optional


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'


class DevEnvironmentSetup:
    """Development environment setup manager."""
    
    def __init__(self, skip_cuda: bool = False, skip_optional: bool = False):
        self.skip_cuda = skip_cuda
        self.skip_optional = skip_optional
        self.project_root = Path(__file__).parent.parent
        self.errors = []
        
    def run_command(self, command: List[str], description: str, 
                   check: bool = True, capture_output: bool = False) -> Optional[subprocess.CompletedProcess]:
        """Run a command with error handling."""
        print(f"{Colors.BLUE}🔄 {description}...{Colors.END}")
        
        try:
            result = subprocess.run(
                command,
                cwd=self.project_root,
                check=check,
                capture_output=capture_output,
                text=True
            )
            
            if result.returncode == 0:
                print(f"{Colors.GREEN}✅ {description} completed successfully{Colors.END}")
            else:
                print(f"{Colors.YELLOW}⚠️ {description} completed with warnings{Colors.END}")
                
            return result
            
        except subprocess.CalledProcessError as e:
            print(f"{Colors.RED}❌ {description} failed: {e}{Colors.END}")
            if capture_output and e.stderr:
                print(f"Error output: {e.stderr}")
            self.errors.append(f"{description}: {str(e)}")
            return None
        except FileNotFoundError as e:
            print(f"{Colors.RED}❌ Command not found: {e}{Colors.END}")
            self.errors.append(f"{description}: Command not found")
            return None
    
    def check_python_version(self) -> bool:
        """Check Python version compatibility."""
        version = sys.version_info
        if version.major == 3 and version.minor >= 8:
            print(f"{Colors.GREEN}✅ Python {version.major}.{version.minor}.{version.micro} is compatible{Colors.END}")
            return True
        else:
            print(f"{Colors.RED}❌ Python {version.major}.{version.minor}.{version.micro} is not supported{Colors.END}")
            print(f"{Colors.CYAN}Please install Python 3.8 or higher{Colors.END}")
            return False
    
    def detect_package_manager(self) -> str:
        """Detect available Python package manager."""
        managers = ['conda', 'poetry', 'pip']
        
        for manager in managers:
            try:
                result = subprocess.run(
                    [manager, '--version'], 
                    capture_output=True, 
                    text=True,
                    check=True
                )
                print(f"{Colors.GREEN}✅ Detected {manager}: {result.stdout.strip()}{Colors.END}")
                return manager
            except (subprocess.CalledProcessError, FileNotFoundError):
                continue
        
        print(f"{Colors.RED}❌ No supported package manager found{Colors.END}")
        return "pip"  # Fallback to pip
    
    def setup_virtual_environment(self, package_manager: str) -> bool:
        """Set up virtual environment."""
        if package_manager == "conda":
            # Check if conda environment already exists
            result = self.run_command(
                ["conda", "env", "list"],
                "Checking conda environments",
                check=False,
                capture_output=True
            )
            
            if result and "aetherist" not in result.stdout:
                # Create new conda environment
                self.run_command(
                    ["conda", "create", "-n", "aetherist", "python=3.9", "-y"],
                    "Creating conda environment 'aetherist'"
                )
                
                print(f"{Colors.CYAN}Activate with: conda activate aetherist{Colors.END}")
            else:
                print(f"{Colors.GREEN}✅ Conda environment 'aetherist' already exists{Colors.END}")
                
        elif package_manager == "poetry":
            # Check if poetry project is initialized
            if not (self.project_root / "pyproject.toml").exists():
                self.run_command(
                    ["poetry", "init", "--no-interaction"],
                    "Initializing Poetry project"
                )
            
            # Create virtual environment
            self.run_command(
                ["poetry", "env", "use", "python"],
                "Setting up Poetry virtual environment"
            )
            
        else:  # pip
            venv_path = self.project_root / "venv"
            if not venv_path.exists():
                self.run_command(
                    [sys.executable, "-m", "venv", "venv"],
                    "Creating Python virtual environment"
                )
                
                if platform.system() == "Windows":
                    activate_script = venv_path / "Scripts" / "activate"
                else:
                    activate_script = venv_path / "bin" / "activate"
                    
                print(f"{Colors.CYAN}Activate with: source {activate_script}{Colors.END}")
            else:
                print(f"{Colors.GREEN}✅ Virtual environment already exists{Colors.END}")
        
        return True
    
    def install_pytorch(self, package_manager: str) -> bool:
        """Install PyTorch with appropriate CUDA support."""
        if self.skip_cuda:
            pytorch_package = "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
            description = "Installing PyTorch (CPU only)"
        else:
            # Detect CUDA availability
            try:
                result = subprocess.run(
                    ["nvidia-smi"], 
                    capture_output=True, 
                    check=True
                )
                print(f"{Colors.GREEN}✅ NVIDIA GPU detected{Colors.END}")
                pytorch_package = "torch torchvision torchaudio"
                description = "Installing PyTorch with CUDA support"
            except (subprocess.CalledProcessError, FileNotFoundError):
                print(f"{Colors.YELLOW}⚠️ No NVIDIA GPU detected, installing CPU version{Colors.END}")
                pytorch_package = "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
                description = "Installing PyTorch (CPU only)"
        
        if package_manager == "conda":
            self.run_command(
                ["conda", "install", "-n", "aetherist", "-c", "pytorch"] + pytorch_package.split(),
                description
            )
        else:
            self.run_command(
                ["pip", "install"] + pytorch_package.split(),
                description
            )
        
        return True
    
    def install_requirements(self, package_manager: str) -> bool:
        """Install project requirements."""
        requirements_files = [
            "requirements.txt",
            "requirements-dev.txt" if not self.skip_optional else None
        ]
        
        for req_file in requirements_files:
            if req_file is None:
                continue
                
            req_path = self.project_root / req_file
            if req_path.exists():
                if package_manager == "conda":
                    self.run_command(
                        ["conda", "install", "-n", "aetherist", "--file", str(req_path)],
                        f"Installing {req_file} with conda",
                        check=False  # Some packages might not be available in conda
                    )
                    # Fallback to pip for unavailable packages
                    self.run_command(
                        ["pip", "install", "-r", str(req_path)],
                        f"Installing remaining packages from {req_file} with pip"
                    )
                elif package_manager == "poetry":
                    # Convert requirements.txt to poetry dependencies (simplified)
                    self.run_command(
                        ["pip", "install", "-r", str(req_path)],
                        f"Installing {req_file} (Poetry fallback to pip)"
                    )
                else:
                    self.run_command(
                        ["pip", "install", "-r", str(req_path)],
                        f"Installing {req_file}"
                    )
        
        return True
    
    def install_package_editable(self, package_manager: str) -> bool:
        """Install the package in editable mode."""
        if package_manager == "poetry":
            self.run_command(
                ["poetry", "install"],
                "Installing Aetherist in editable mode with Poetry"
            )
        else:
            self.run_command(
                ["pip", "install", "-e", "."],
                "Installing Aetherist in editable mode"
            )
        
        return True
    
    def setup_pre_commit_hooks(self) -> bool:
        """Set up pre-commit hooks for development."""
        if self.skip_optional:
            return True
            
        # Install pre-commit if not already installed
        self.run_command(
            ["pip", "install", "pre-commit"],
            "Installing pre-commit"
        )
        
        # Create .pre-commit-config.yaml if it doesn't exist
        precommit_config = self.project_root / ".pre-commit-config.yaml"
        if not precommit_config.exists():
            config_content = """repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
      - id: check-merge-conflict
      - id: debug-statements

  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
        language_version: python3

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: ["--profile", "black"]

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: ["--max-line-length=88", "--extend-ignore=E203,W503"]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.3.0
    hooks:
      - id: mypy
        additional_dependencies: [types-requests, types-PyYAML]
"""
            
            with open(precommit_config, 'w') as f:
                f.write(config_content)
            
            print(f"{Colors.GREEN}✅ Created .pre-commit-config.yaml{Colors.END}")
        
        # Install pre-commit hooks
        self.run_command(
            ["pre-commit", "install"],
            "Installing pre-commit hooks"
        )
        
        return True
    
    def create_dev_scripts(self) -> bool:
        """Create useful development scripts."""
        scripts_dir = self.project_root / "scripts"
        scripts_dir.mkdir(exist_ok=True)
        
        # Create run_tests.py script
        test_script = scripts_dir / "run_tests.py"
        if not test_script.exists():
            test_content = '''#!/usr/bin/env python3
"""Run comprehensive tests for Aetherist."""

import subprocess
import sys
from pathlib import Path

def run_tests():
    """Run all tests with coverage."""
    project_root = Path(__file__).parent.parent
    
    commands = [
        ["python", "-m", "pytest", "tests/", "-v", "--cov=aetherist", "--cov-report=html"],
        ["python", "-m", "flake8", "aetherist/"],
        ["python", "-m", "black", "--check", "aetherist/"],
        ["python", "-m", "isort", "--check-only", "aetherist/"],
        ["python", "-m", "mypy", "aetherist/"]
    ]
    
    for cmd in commands:
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=project_root)
        if result.returncode != 0:
            print(f"Command failed: {' '.join(cmd)}")
            sys.exit(1)
    
    print("All tests passed!")

if __name__ == "__main__":
    run_tests()
'''
            with open(test_script, 'w') as f:
                f.write(test_content)
            test_script.chmod(0o755)
        
        # Create format_code.py script
        format_script = scripts_dir / "format_code.py"
        if not format_script.exists():
            format_content = '''#!/usr/bin/env python3
"""Format all Python code in the project."""

import subprocess
from pathlib import Path

def format_code():
    """Format code using black and isort."""
    project_root = Path(__file__).parent.parent
    
    commands = [
        ["python", "-m", "black", "aetherist/", "scripts/", "tests/"],
        ["python", "-m", "isort", "aetherist/", "scripts/", "tests/"]
    ]
    
    for cmd in commands:
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, cwd=project_root)
    
    print("Code formatting complete!")

if __name__ == "__main__":
    format_code()
'''
            with open(format_script, 'w') as f:
                f.write(format_content)
            format_script.chmod(0o755)
        
        return True
    
    def setup_vscode_settings(self) -> bool:
        """Create VS Code development settings."""
        if self.skip_optional:
            return True
            
        vscode_dir = self.project_root / ".vscode"
        vscode_dir.mkdir(exist_ok=True)
        
        # Create settings.json
        settings_file = vscode_dir / "settings.json"
        if not settings_file.exists():
            settings_content = '''{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.formatting.provider": "black",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.linting.mypyEnabled": true,
    "python.sortImports.args": ["--profile", "black"],
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    },
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true,
        ".mypy_cache": true,
        ".pytest_cache": true,
        "htmlcov": true
    }
}'''
            with open(settings_file, 'w') as f:
                f.write(settings_content)
        
        # Create launch.json for debugging
        launch_file = vscode_dir / "launch.json"
        if not launch_file.exists():
            launch_content = '''{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Current File",
            "type": "python",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}"
        },
        {
            "name": "Python: Run Tests",
            "type": "python",
            "request": "launch",
            "module": "pytest",
            "args": ["tests/", "-v"],
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}"
        },
        {
            "name": "Python: Aetherist API",
            "type": "python",
            "request": "launch",
            "module": "uvicorn",
            "args": ["aetherist.api.main:app", "--reload"],
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}"
        }
    ]
}'''
            with open(launch_file, 'w') as f:
                f.write(launch_content)
        
        return True
    
    def print_summary(self) -> None:
        """Print setup summary and next steps."""
        print(f"\n{Colors.BOLD}{Colors.BLUE}🎉 Development Environment Setup Complete!{Colors.END}")
        print("=" * 60)
        
        if self.errors:
            print(f"\n{Colors.RED}❌ Some errors occurred during setup:{Colors.END}")
            for error in self.errors:
                print(f"  • {error}")
            print(f"\n{Colors.CYAN}Please review and fix these issues before continuing.{Colors.END}")
        else:
            print(f"{Colors.GREEN}✅ All components installed successfully!{Colors.END}")
        
        print(f"\n{Colors.BOLD}Next Steps:{Colors.END}")
        print(f"{Colors.CYAN}1. Activate your environment:{Colors.END}")
        print("   conda activate aetherist  # if using conda")
        print("   source venv/bin/activate   # if using venv")
        print("   poetry shell               # if using poetry")
        
        print(f"\n{Colors.CYAN}2. Verify installation:{Colors.END}")
        print("   python scripts/verify_installation.py")
        
        print(f"\n{Colors.CYAN}3. Run tests:{Colors.END}")
        print("   python scripts/run_tests.py")
        
        print(f"\n{Colors.CYAN}4. Format code:{Colors.END}")
        print("   python scripts/format_code.py")
        
        print(f"\n{Colors.CYAN}5. Start developing:{Colors.END}")
        print("   python -m aetherist.api.main  # Start API server")
        print("   python examples/basic_usage.py  # Run examples")
        
        print(f"\n{Colors.YELLOW}📚 Documentation:{Colors.END}")
        print("   docs/installation.md     - Installation guide")
        print("   docs/contributing.md     - Contributing guide")
        print("   docs/api_reference.md    - API documentation")
    
    def setup(self) -> bool:
        """Run complete development environment setup."""
        print(f"{Colors.BOLD}{Colors.PURPLE}🚀 Aetherist Development Environment Setup{Colors.END}")
        print("=" * 60)
        print(f"Platform: {platform.platform()}")
        print(f"Python: {sys.version}")
        print()
        
        # Check Python version
        if not self.check_python_version():
            return False
        
        # Detect package manager
        package_manager = self.detect_package_manager()
        
        # Setup steps
        steps = [
            ("Setting up virtual environment", lambda: self.setup_virtual_environment(package_manager)),
            ("Installing PyTorch", lambda: self.install_pytorch(package_manager)),
            ("Installing requirements", lambda: self.install_requirements(package_manager)),
            ("Installing Aetherist (editable)", lambda: self.install_package_editable(package_manager)),
            ("Setting up pre-commit hooks", self.setup_pre_commit_hooks),
            ("Creating development scripts", self.create_dev_scripts),
            ("Configuring VS Code settings", self.setup_vscode_settings),
        ]
        
        for step_name, step_func in steps:
            print(f"\n{Colors.BOLD}📦 {step_name}:{Colors.END}")
            try:
                step_func()
            except Exception as e:
                print(f"{Colors.RED}❌ {step_name} failed: {e}{Colors.END}")
                self.errors.append(f"{step_name}: {str(e)}")
        
        self.print_summary()
        return len(self.errors) == 0


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(
        description="Set up Aetherist development environment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script will:
1. Create a virtual environment
2. Install PyTorch with CUDA support (if available)
3. Install all dependencies
4. Set up pre-commit hooks
5. Configure development tools
6. Create useful scripts

Examples:
  python scripts/setup_dev_env.py                 # Full setup
  python scripts/setup_dev_env.py --skip-cuda     # CPU-only PyTorch
  python scripts/setup_dev_env.py --skip-optional # Minimal setup
        """
    )
    
    parser.add_argument(
        "--skip-cuda",
        action="store_true",
        help="Install CPU-only PyTorch (skip CUDA detection)"
    )
    
    parser.add_argument(
        "--skip-optional", 
        action="store_true",
        help="Skip optional development tools (pre-commit, VS Code config)"
    )
    
    args = parser.parse_args()
    
    # Run setup
    setup = DevEnvironmentSetup(
        skip_cuda=args.skip_cuda,
        skip_optional=args.skip_optional
    )
    
    success = setup.setup()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()