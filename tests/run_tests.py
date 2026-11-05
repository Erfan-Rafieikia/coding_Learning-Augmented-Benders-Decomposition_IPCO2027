#!/usr/bin/env python3
"""
Test runner for ML-augmented Benders decomposition framework.

This script provides a convenient way to run tests with different configurations
and generate coverage reports.
"""

import subprocess
import sys
import os
import argparse
from pathlib import Path


def find_python_executable():
    """Find the appropriate Python executable for running tests."""
    # Check if we're in the project directory
    project_root = Path(__file__).parent
    
    # Check for virtual environment
    venv_paths = [
        project_root / "bd_ml" / "Scripts" / "python.exe",  # Windows venv
        project_root / "bd_ml" / "bin" / "python",         # Unix venv
        project_root / "venv" / "Scripts" / "python.exe",  # Windows venv (alternative)
        project_root / "venv" / "bin" / "python",          # Unix venv (alternative)
    ]
    
    for venv_python in venv_paths:
        if venv_python.exists():
            return str(venv_python)
    
    # Fall back to system Python
    return sys.executable


def run_tests(test_type="all", coverage=False, verbose=False, parallel=False):
    """
    Run tests with specified configuration.
    
    Args:
        test_type (str): Type of tests to run ('all', 'unit', 'integration', 'fast')
        coverage (bool): Generate coverage report
        verbose (bool): Verbose output
        parallel (bool): Run tests in parallel
    """
    
    # Base pytest command
    python_exe = find_python_executable()
    cmd = [python_exe, "-m", "pytest"]
    
    # Add test selection
    if test_type == "unit":
        cmd.extend(["-m", "unit"])
    elif test_type == "integration":
        cmd.extend(["-m", "integration"])
    elif test_type == "fast":
        cmd.extend(["-m", "not slow"])
    # 'all' runs everything by default
    
    # Add coverage if requested
    if coverage:
        cmd.extend(["--cov=src", "--cov-report=html", "--cov-report=term"])
    
    # Add verbosity
    if verbose:
        cmd.append("-v")
    else:
        cmd.append("-q")
    
    # Add parallel execution
    if parallel:
        cmd.extend(["-n", "auto"])
    
    # Add test directory
    cmd.append("tests/")
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except FileNotFoundError:
        print("Error: pytest not found. Install with: pip install -r requirements-test.txt")
        return 1


def check_code_quality():
    """Run code quality checks."""
    python_exe = find_python_executable()
    
    quality_checks = [
        ([python_exe, "-m", "flake8", "src/", "tests/"], "Flake8 style check"),
        ([python_exe, "-m", "black", "--check", "src/", "tests/"], "Black formatting check"),
        ([python_exe, "-m", "isort", "--check", "src/", "tests/"], "Import sorting check"),
    ]
    
    all_passed = True
    
    for cmd, description in quality_checks:
        print(f"\nRunning {description}...")
        try:
            result = subprocess.run(cmd, check=False, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✓ {description} passed")
            else:
                print(f"✗ {description} failed")
                print(result.stdout)
                print(result.stderr)
                all_passed = False
        except FileNotFoundError:
            print(f"⚠ {description} skipped (tool not installed)")
    
    return all_passed


def main():
    """Main test runner function."""
    
    parser = argparse.ArgumentParser(description="Test runner for Benders decomposition framework")
    
    parser.add_argument(
        "--type", 
        choices=["all", "unit", "integration", "fast"], 
        default="all",
        help="Type of tests to run"
    )
    
    parser.add_argument(
        "--coverage", 
        action="store_true",
        help="Generate coverage report"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true", 
        help="Verbose output"
    )
    
    parser.add_argument(
        "--parallel", "-j",
        action="store_true",
        help="Run tests in parallel"
    )
    
    parser.add_argument(
        "--quality",
        action="store_true",
        help="Run code quality checks"
    )
    
    args = parser.parse_args()
    
    # Change to project directory
    project_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_dir)
    
    exit_code = 0
    
    # Run code quality checks if requested
    if args.quality:
        print("=" * 60)
        print("Running Code Quality Checks")
        print("=" * 60)
        
        if not check_code_quality():
            exit_code = 1
    
    # Run tests
    print("=" * 60)
    print(f"Running Tests: {args.type}")
    print("=" * 60)
    
    test_exit_code = run_tests(
        test_type=args.type,
        coverage=args.coverage, 
        verbose=args.verbose,
        parallel=args.parallel
    )
    
    if test_exit_code != 0:
        exit_code = test_exit_code
    
    # Summary
    print("\n" + "=" * 60)
    if exit_code == 0:
        print("✅ All checks passed!")
    else:
        print("❌ Some checks failed!")
    print("=" * 60)
    
    if args.coverage:
        print("\n📊 Coverage report generated in htmlcov/index.html")
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
