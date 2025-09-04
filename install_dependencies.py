"""
Dependency Installer for Ronin V2.0 Trading Bot
Installs all required Python packages for the V2.0 strategy.
"""

import subprocess
import sys
import os

def install_package(package):
    """Install a package using pip"""
    try:
        print(f"📦 Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ {package} installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package}: {e}")
        return False

def main():
    """Install all required dependencies"""
    print("🚀 Installing Ronin V2.0 Dependencies")
    print("=" * 50)
    
    # Core dependencies (essential)
    core_packages = [
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "yfinance>=0.2.18",
        "pytz>=2023.3",
        "matplotlib>=3.7.0",
        "aiohttp>=3.8.0",
        "requests>=2.31.0"
    ]
    
    # Optional dependencies
    optional_packages = [
        "scipy>=1.10.0",
        "seaborn>=0.12.0",
        "MetaTrader5>=5.0.45",
        "alpaca-trade-api>=3.0.0",
        "websockets>=11.0.0",
        "asyncio-mqtt>=0.13.0",
        "pyyaml>=6.0",
        "python-dotenv>=1.0.0"
    ]
    
    # Install core packages first
    print("\n📋 Installing Core Dependencies...")
    failed_core = []
    for package in core_packages:
        if not install_package(package):
            failed_core.append(package)
    
    # Install optional packages
    print("\n📋 Installing Optional Dependencies...")
    failed_optional = []
    for package in optional_packages:
        if not install_package(package):
            failed_optional.append(package)
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Installation Summary:")
    
    if not failed_core:
        print("✅ All core dependencies installed successfully!")
    else:
        print(f"❌ Failed core dependencies: {failed_core}")
    
    if failed_optional:
        print(f"⚠️  Failed optional dependencies: {failed_optional}")
        print("   (These are not critical for basic functionality)")
    
    # Test imports
    print("\n🔍 Testing Core Imports...")
    test_imports = [
        ("pandas", "pd"),
        ("numpy", "np"),
        ("yfinance", "yf"),
        ("matplotlib", "plt"),
        ("pytz", None)
    ]
    
    import_failures = []
    for module, alias in test_imports:
        try:
            if alias:
                exec(f"import {module} as {alias}")
            else:
                exec(f"import {module}")
            print(f"✅ {module} import successful")
        except ImportError as e:
            print(f"❌ {module} import failed: {e}")
            import_failures.append(module)
    
    print("\n" + "=" * 50)
    if not import_failures:
        print("🎉 All core dependencies are working!")
        print("You can now run Ronin V2.0 strategy.")
        print("\nNext steps:")
        print("1. python cli.py config --strategy v2")
        print("2. python cli.py backtest --symbol NVDA --strategy v2")
    else:
        print(f"❌ Import failures: {import_failures}")
        print("Please install missing dependencies manually:")
        for module in import_failures:
            print(f"   pip install {module}")

if __name__ == "__main__":
    main()
