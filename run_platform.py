import sys
import subprocess
import time
import signal
import os

def main():
    print("======================================================================")
    # Human-readable model information
    print("STARTING MULTIMODAL AI WELFARE MONITORING DASHBOARD & API BACKEND")
    print("======================================================================")
    
    # Virtual environment paths
    venv_bin = "Scripts" if sys.platform == "win32" else "bin"
    python_exe = os.path.join(".venv", venv_bin, "python")
    uvicorn_exe = os.path.join(".venv", venv_bin, "uvicorn")
    streamlit_exe = os.path.join(".venv", venv_bin, "streamlit")
    
    if not os.path.exists(python_exe):
        # Fallback to system executables if virtual env is not local
        python_exe = "python"
        uvicorn_exe = "uvicorn"
        streamlit_exe = "streamlit"

    processes = []
    try:
        # 1. Start the FastAPI backend
        print("[System] Launching FastAPI Backend on http://127.0.0.1:8000...")
        api_proc = subprocess.Popen(
            [python_exe, "-m", "uvicorn", "api_server:app", "--host", "127.0.0.1", "--port", "8000"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        processes.append(api_proc)
        
        # Give API server 2 seconds to bind port
        time.sleep(2.0)
        
        # 2. Start Streamlit Dashboard
        print("[System] Launching Streamlit Dashboard on http://localhost:8501...")
        dashboard_proc = subprocess.Popen(
            [python_exe, "-m", "streamlit", "run", "dashboard.py", "--server.port", "8501"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        processes.append(dashboard_proc)
        
        print("\n[System] Both services are running successfully. Press Ctrl+C to terminate both.")
        
        # Monitor processes and print outputs
        while True:
            # Check if any process terminated
            for p in processes:
                if p.poll() is not None:
                    print(f"\n[System] Process {p.pid} terminated with exit code {p.returncode}")
                    return
            time.sleep(1.0)
            
    except KeyboardInterrupt:
        print("\n[System] Terminating services...")
    finally:
        for p in processes:
            if p.poll() is None:
                p.terminate()
                p.wait()
        print("[System] All services shut down successfully.")

if __name__ == "__main__":
    main()
