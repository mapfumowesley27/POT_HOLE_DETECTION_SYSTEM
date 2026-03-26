import subprocess
import time
import webbrowser
import os
import sys

def start_backend():
    print("🚀 Starting Backend...")
    # Navigate to the backend directory and run run.py
    # We use sys.executable to    ensure we use the same Python interpreter
    backend_script = os.path.join("backend", "run.py")
    # Setting PYTHONPATH so it can find 'app'
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.join(os.getcwd(), "backend")
    
    return subprocess.Popen([sys.executable, backend_script], env=env)

def start_frontend():
    print("🌐 Starting Frontend...")
    # Serve the 'frontend' directory using Python's built-in http.server on port 8000
    frontend_dir = os.path.abspath("frontend")
    
    # We use Popen to run the HTTP server in the background
    return subprocess.Popen(
        [sys.executable, "-m", "http.server", "8000"],
        cwd=frontend_dir
    )

def main():
    backend_proc = None
    frontend_proc = None
    try:
        # Start backend
        backend_proc = start_backend()
        
        # Start frontend server
        frontend_proc = start_frontend()
        
        # Give them a moment to start
        print("⏳ Waiting for servers to initialize...")
        time.sleep(3)
        
        # Open the frontend in the default browser
        print("🌍 Opening application in your browser...")
        webbrowser.open("http://localhost:8000/login.html")
        
        print("\n✅ System is running!")
        print("Backend: http://localhost:5000 (API)")
        print("Frontend: http://localhost:8000")
        print("\nPress Ctrl+C to stop both servers.")
        
        # Keep the script running
        while True:
            time.sleep(1)
            # Check if processes are still running
            if backend_proc.poll() is not None:
                print("⚠️ Backend stopped unexpectedly.")
                break
            if frontend_proc.poll() is not None:
                print("⚠️ Frontend stopped unexpectedly.")
                break
                
    except KeyboardInterrupt:
        print("\n🛑 Stopping servers...")
    except Exception as e:
        print(f"❌ Error occurred: {e}")
    finally:
        if backend_proc:
            backend_proc.terminate()
            print("Backend stopped.")
        if frontend_proc:
            frontend_proc.terminate()
            print("Frontend stopped.")
        print("👋 Goodbye!")

if __name__ == "__main__":
    main()
