#!/usr/bin/env python3
"""
CLI Commands Verification Test (Parameterized version)

This test verifies that all CLI commands work correctly when the package is 
built and installed using pyproject.toml. It's part of the migration strategy 
described in issue #1415: https://github.com/clearml/clearml/issues/1415

The test:
1. Creates an isolated virtual environment
2. Builds the package with pyproject.toml
3. Installs the built wheel in the virtual environment
4. Runs each CLI command with --help to verify they're properly installed and working

This version uses pytest's parametrize to run each command as a separate test.
"""

import pytest
import sys
import tempfile
import subprocess
import venv
from pathlib import Path
import shutil


def run_command(cmd, cwd=None, exit_on_error=False):
    """Run a command and return its output."""
    cmd_str = ' '.join(str(arg) for arg in cmd)
    print(f"Running: {cmd_str}")
    try:
        result = subprocess.run(
            cmd, cwd=cwd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"\nCommand failed with exit code {e.returncode}")
        print(f"\nSTDOUT:\n{e.stdout}")
        print(f"\nSTDERR:\n{e.stderr}")
        if exit_on_error:
            pytest.fail(f"Command {cmd_str} failed with exit code {e.returncode}")
        raise


@pytest.fixture(scope="module")
def cli_environment():
    """Create a temporary environment for testing CLI commands."""
    # CLI commands to verify
    cli_commands = [
        "clearml-init",
        "clearml-data",
        "clearml-task",
        "clearml-param-search",
        "clearml-debug"
    ]
    
    # Create temporary directories
    temp_dir = Path(tempfile.mkdtemp(prefix="clearml_cli_verify_"))
    venv_path = temp_dir / "venv"
    
    # Track artifacts to clean up
    project_root = Path.cwd()
    artifacts_to_clean = [
        project_root / "dist",
        project_root / "build"
    ]
    
    # Create virtual environment
    print(f"Creating virtual environment at {venv_path}...")
    venv.create(venv_path, with_pip=True)
    
    # Get the Python executable path for this venv
    if sys.platform == 'win32':
        python_exe = venv_path / 'Scripts' / 'python.exe'
    else:
        python_exe = venv_path / 'bin' / 'python'
    
    # Install required packages for building
    print("Installing required packages...")
    subprocess.check_call([str(python_exe), '-m', 'pip', 'install', '--upgrade', 'pip'])
    subprocess.check_call([str(python_exe), '-m', 'pip', 'install', 'build'])
    
    # Build with pyproject.toml
    print("\n=== Building with pyproject.toml ===")
    run_command([str(python_exe), "-m", "build"], 
                cwd=project_root, exit_on_error=True)
    
    # Find the wheel
    wheel_file = next(Path('dist').glob('*.whl'))
    print(f"Found wheel: {wheel_file}")
    
    # Install the wheel
    print("\n=== Installing wheel in isolated environment ===")
    run_command([str(python_exe), "-m", "pip", "install", str(wheel_file)],
                exit_on_error=True)
    
    # Create a structure to return the environment variables
    env = {
        'temp_dir': temp_dir,
        'venv_path': venv_path,
        'python_exe': python_exe,
        'project_root': project_root,
        'artifacts_to_clean': artifacts_to_clean,
        'cli_commands': cli_commands
    }
    
    yield env
    
    # Clean up
    print(f"\nCleaning up temporary directory: {temp_dir}")
    shutil.rmtree(temp_dir)
    
    # Clean up build artifacts
    for path in artifacts_to_clean:
        if path.exists():
            print(f"Cleaning up: {path}")
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()


@pytest.mark.parametrize("cmd_name", [
    "clearml-init",
    "clearml-data",
    "clearml-task",
    "clearml-param-search",
    "clearml-debug"
])
def test_cli_command(cli_environment, cmd_name):
    """Test each CLI command individually."""
    env = cli_environment
    
    # Get the path to the command in the virtual environment
    if sys.platform == 'win32':
        cmd_path = env['venv_path'] / 'Scripts' / f"{cmd_name}.exe"
    else:
        cmd_path = env['venv_path'] / 'bin' / cmd_name
    
    # Check if the command exists
    assert cmd_path.exists(), f"Command not found: {cmd_name}"
            
    print(f"Verifying command: {cmd_name}")
    # Run the command with --help and verify it succeeds
    output = run_command([str(cmd_path), "--help"], exit_on_error=True)
    print(f"✅ Command {cmd_name} verified successfully")
    
    # Assertions about the output
    assert "usage:" in output, f"Command {cmd_name} help output doesn't look right"