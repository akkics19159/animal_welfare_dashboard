#!/usr/bin/env python
"""
Dependency checker for the Animal Welfare Dashboard.
Validates all required libraries and system tools (FFmpeg).
"""

import sys
import subprocess
import shutil


def check_python_package(package_name, import_name=None):
    """Check if a Python package is installed."""
    import_name = import_name or package_name
    try:
        __import__(import_name)
        return True, None
    except ImportError as e:
        return False, str(e)


def check_ffmpeg():
    """Check if FFmpeg is installed and accessible via PATH."""
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        try:
            result = subprocess.run(
                [ffmpeg_path, "-version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                # Extract version line
                first_line = result.stdout.split("\n")[0]
                return True, f"Found at {ffmpeg_path}: {first_line}"
        except Exception as e:
            return False, f"FFmpeg found but failed to execute: {e}"
    return False, "FFmpeg not found in PATH"


def check_audio_backends():
    """Check if audio backend libraries are available."""
    backends = {
        "pysoundfile": "soundfile",
        "audioread": "audioread",
        "librosa": "librosa",
    }
    results = {}
    for pkg, import_name in backends.items():
        installed, error = check_python_package(pkg, import_name)
        results[import_name] = (installed, error)
    return results


def main():
    print("=" * 70)
    print("ANIMAL WELFARE DASHBOARD - DEPENDENCY CHECK")
    print("=" * 70)

    # Python version
    print(f"\n[Python Version]")
    print(f"  {sys.version}")

    # Required packages
    required_packages = {
        "numpy": "numpy",
        "opencv-python": "cv2",
        "librosa": "librosa",
        "sounddevice": "sounddevice",
        "streamlit": "streamlit",
        "pandas": "pandas",
        "ultralytics": "ultralytics",
    }

    print(f"\n[Required Python Packages]")
    all_required_ok = True
    for pkg, import_name in required_packages.items():
        installed, error = check_python_package(pkg, import_name)
        status = "[OK] INSTALLED" if installed else "[FAIL] MISSING"
        print(f"  {pkg:<25} {status}")
        if not installed:
            all_required_ok = False
            print(f"    -> {error}")

    # Optional packages
    optional_packages = {
        "moviepy": "moviepy",
    }

    print(f"\n[Optional Python Packages]")
    for pkg, import_name in optional_packages.items():
        installed, error = check_python_package(pkg, import_name)
        status = "[OK] INSTALLED" if installed else "[-] NOT INSTALLED (optional)"
        print(f"  {pkg:<25} {status}")
        if not installed and error:
            print(f"    -> {error}")

    # Audio backends
    print(f"\n[Audio Backend Libraries]")
    audio_backends = check_audio_backends()
    for backend, (installed, error) in audio_backends.items():
        status = "[OK] AVAILABLE" if installed else "[-] NOT AVAILABLE"
        print(f"  {backend:<25} {status}")

    # FFmpeg system tool
    print(f"\n[System Tools]")
    ffmpeg_ok, ffmpeg_msg = check_ffmpeg()
    ffmpeg_status = "[OK] AVAILABLE" if ffmpeg_ok else "[FAIL] REQUIRED BUT MISSING"
    print(f"  ffmpeg                  {ffmpeg_status}")
    print(f"    -> {ffmpeg_msg}")

    # Summary
    print(f"\n" + "=" * 70)
    if all_required_ok and ffmpeg_ok:
        print("[OK] ALL CRITICAL DEPENDENCIES MET - Ready to use!")
        print("=" * 70)
        return 0
    else:
        print("[FAIL] MISSING CRITICAL DEPENDENCIES")
        print("=" * 70)
        if not ffmpeg_ok:
            print("\n[ACTION REQUIRED: Install FFmpeg]")
            print("  FFmpeg is required for robust audio extraction.")
            print("  ")
            print("  Windows: Download from https://www.gyan.dev/ffmpeg/builds/")
            print("           Extract and add bin/ folder to PATH")
            print("  macOS:   brew install ffmpeg")
            print("  Linux:   sudo apt-get install ffmpeg")
        if not all_required_ok:
            print("\n[ACTION REQUIRED: Install missing Python packages]")
            print("  Run: pip install -r requirements.txt")
        return 1


if __name__ == "__main__":
    sys.exit(main())
