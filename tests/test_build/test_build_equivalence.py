#!/usr/bin/env python3
"""
Build Equivalence Verification Test

This test verifies that building with pyproject.toml produces the same results 
as building with setup.py. It's part of the migration strategy described in 
issue #1415: https://github.com/clearml/clearml/issues/1415

The test:
1. Creates an isolated virtual environment
2. Builds the package with setup.py
3. Builds the package with pyproject.toml
4. Compares the contents of both wheels and sdists

This helps ensure that the dual build system configuration (supporting both 
setuptools via setup.py and hatchling via pyproject.toml) maintains compatibility 
during the migration period.
"""

import pytest
import sys
import hashlib
import tempfile
import subprocess
import venv
from pathlib import Path
import zipfile
import tarfile
import shutil
from collections import defaultdict


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


def get_file_hash(filepath):
    """Calculate the SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


@pytest.fixture
def build_environment():
    """Create a temporary environment for building packages."""
    # Create temporary directories for builds
    temp_dir = Path(tempfile.mkdtemp(prefix="clearml_migration_test_"))
    venv_path = temp_dir / "venv"
    setuppy_dir = temp_dir / "setup_py_build"
    pyproject_dir = temp_dir / "pyproject_build"
    
    # Create output directories
    setuppy_dir.mkdir(parents=True, exist_ok=True)
    pyproject_dir.mkdir(parents=True, exist_ok=True)
    
    # Track artifacts to clean up
    project_root = Path.cwd()
    artifacts_to_clean = [
        project_root / "dist",
        project_root / "build",
        project_root / "clearml.egg-info"
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
    subprocess.check_call([str(python_exe), '-m', 'pip', 'install', 'wheel', 'build', 'setuptools'])
    
    # Also install requirements for clearml to build successfully
    print("Installing build dependencies (from requirements.txt)...")
    try:
        requirements_file = project_root / "requirements.txt"
        if requirements_file.exists():
            subprocess.check_call([
                str(python_exe), '-m', 'pip', 'install', '-r', str(requirements_file)
            ])
    except Exception as e:
        print(f"Warning: Failed to install requirements.txt: {e}")
    
    # Create a structure to return the environment variables
    env = {
        'temp_dir': temp_dir,
        'venv_path': venv_path,
        'setuppy_dir': setuppy_dir,
        'pyproject_dir': pyproject_dir,
        'python_exe': python_exe,
        'project_root': project_root,
        'artifacts_to_clean': artifacts_to_clean
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


# Define known differences that are expected and should be ignored
EXPECTED_WHEEL_DIFFERENCES = [
    # Metadata directory created by the build system
    '.dist-info/',
    # Files that always have different content due to build system differences
    '.dist-info/RECORD',
    '.dist-info/WHEEL',
    '.dist-info/METADATA',
    '.dist-info/top_level.txt'
]

EXPECTED_SDIST_DIFFERENCES = [
    # Build system files
    'PKG-INFO', 
    'setup.cfg', 
    'setup.py', 
    'pyproject.toml',
    # Hatchling always includes .gitignore by design
    '.gitignore',
    # Setuptools-specific metadata directory
    'clearml.egg-info'
]

def list_wheel_files(wheel_path):
    """Extract the list of files in a wheel file."""
    files = {}
    with zipfile.ZipFile(wheel_path, 'r') as wheel:
        for info in wheel.infolist():
            if not info.is_dir():
                # Skip files that will naturally differ or are not relevant for comparison
                if any(info.filename.startswith(diff) or info.filename.endswith(diff) 
                       for diff in EXPECTED_WHEEL_DIFFERENCES) or info.filename.startswith('tests/'):
                    continue
                
                # Use a temporary file to extract and hash the content
                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    tmp_path = Path(tmp.name)
                    tmp.write(wheel.read(info.filename))
                
                file_hash = get_file_hash(tmp_path)
                tmp_path.unlink()  # Delete the temp file
                
                files[info.filename] = {
                    'size': info.file_size,
                    'hash': file_hash
                }
    return files


def list_sdist_files(sdist_path):
    """Extract the list of files in a source distribution."""
    files = {}
    with tarfile.open(sdist_path, 'r:gz') as sdist:
        for member in sdist.getmembers():
            if not member.isdir():
                # Get the member name without the top directory prefix
                path_parts = member.name.split('/', 1)
                if len(path_parts) > 1:
                    relative_path = path_parts[1]
                else:
                    relative_path = member.name
                
                # Skip files that will naturally differ between build methods
                if any(relative_path.startswith(prefix) for prefix in EXPECTED_SDIST_DIFFERENCES) or relative_path.startswith('tests'):
                    continue
                
                f = sdist.extractfile(member)
                if f:
                    content = f.read()
                    file_hash = hashlib.sha256(content).hexdigest()
                    files[relative_path] = {
                        'size': member.size,
                        'hash': file_hash
                    }
    return files


def compare_file_lists(files1, files2, label1="setup.py", label2="pyproject.toml"):
    """Compare two file lists and print the differences."""
    all_files = sorted(set(files1.keys()) | set(files2.keys()))
    
    missing_in_1 = []
    missing_in_2 = []
    different_content = []
    different_size = []
    
    for file in all_files:
        if file not in files1:
            missing_in_1.append(file)
        elif file not in files2:
            missing_in_2.append(file)
        elif files1[file]['hash'] != files2[file]['hash']:
            if files1[file]['size'] != files2[file]['size']:
                different_size.append((file, files1[file]['size'], files2[file]['size']))
            different_content.append(file)
    
    print(f"\nComparing {label1} build vs {label2} build:")
    print(f"Total files: {len(all_files)}")
    
    if missing_in_1:
        print(f"\nFiles missing in {label1} build ({len(missing_in_1)}):")
        for file in missing_in_1:
            print(f"  - {file}")
    
    if missing_in_2:
        print(f"\nFiles missing in {label2} build ({len(missing_in_2)}):")
        for file in missing_in_2:
            print(f"  - {file}")
    
    if different_size:
        print(f"\nFiles with different sizes ({len(different_size)}):")
        for file, size1, size2 in different_size:
            print(f"  - {file}: {size1} bytes in {label1}, {size2} bytes in {label2}")
    
    if different_content and not different_size:
        print(f"\nFiles with same size but different content ({len(different_content)}):")
        for file in different_content:
            if file not in [f for f, _, _ in different_size]:
                print(f"  - {file}")
    
    return bool(missing_in_1 or missing_in_2 or different_content)


def find_distribution_files(directory):
    """Find wheel and sdist files in the given directory."""
    directory = Path(directory)
    wheel_files = list(directory.glob('*.whl'))
    sdist_files = list(directory.glob('*.tar.gz'))
    
    if not wheel_files:
        pytest.fail(f"No wheel files found in {directory}")
    
    if not sdist_files:
        pytest.fail(f"No source distribution files found in {directory}")
    
    return wheel_files[0], sdist_files[0]


def test_build_equivalence(build_environment):
    """Test that both build methods produce equivalent packages."""
    env = build_environment
    
    # Build with setup.py
    print("\n=== Building with setup.py ===")
    run_command([str(env['python_exe']), "setup.py", "sdist", "bdist_wheel"], 
                cwd=env['project_root'], exit_on_error=True)
    
    # Move the built distributions to our temp directory
    dist_dir = env['project_root'] / "dist"
    if dist_dir.exists():
        for dist_file in dist_dir.glob("*"):
            shutil.move(str(dist_file), str(env['setuppy_dir']))
        
        # Clean up the dist directory
        shutil.rmtree(dist_dir)
    
    # Clean up any remaining build artifacts
    build_dir = env['project_root'] / "build"
    if build_dir.exists():
        shutil.rmtree(build_dir)
    
    # Build with pyproject.toml
    print("\n=== Building with pyproject.toml ===")
    run_command([str(env['python_exe']), "-m", "build"], 
                cwd=env['project_root'], exit_on_error=True)
    
    # Move the built distributions to our temp directory
    dist_dir = env['project_root'] / "dist"
    if dist_dir.exists():
        for dist_file in dist_dir.glob("*"):
            shutil.move(str(dist_file), str(env['pyproject_dir']))
    
    # Compare wheel contents
    setuppy_wheel, setuppy_sdist = find_distribution_files(env['setuppy_dir'])
    pyproject_wheel, pyproject_sdist = find_distribution_files(env['pyproject_dir'])
    
    print("\n==== Comparing wheel files ====")
    print(f"setup.py wheel: {setuppy_wheel.name}")
    print(f"pyproject.toml wheel: {pyproject_wheel.name}")
    
    setuppy_wheel_files = list_wheel_files(setuppy_wheel)
    pyproject_wheel_files = list_wheel_files(pyproject_wheel)
    wheel_differences = compare_file_lists(setuppy_wheel_files, pyproject_wheel_files)
    
    # Compare sdist contents
    print("\n==== Comparing source distributions ====")
    print(f"setup.py sdist: {setuppy_sdist.name}")
    print(f"pyproject.toml sdist: {pyproject_sdist.name}")
    
    setuppy_sdist_files = list_sdist_files(setuppy_sdist)
    pyproject_sdist_files = list_sdist_files(pyproject_sdist)
    sdist_differences = compare_file_lists(setuppy_sdist_files, pyproject_sdist_files)
    
    # Assert no differences
    assert not wheel_differences, "Wheel packages are not equivalent - see output above for details"
    assert not sdist_differences, "Source distributions are not equivalent - see output above for details"
    
    # If everything passed, print a success message
    print("\n✅ Both build methods produce equivalent packages (excluding expected differences)")
    print(f"- Known wheel differences excluded: {', '.join(EXPECTED_WHEEL_DIFFERENCES)}")
    print(f"- Known sdist differences excluded: {', '.join(EXPECTED_SDIST_DIFFERENCES)}")
    print("- Test files excluded from comparison")