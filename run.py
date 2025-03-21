import subprocess
import sys
import time
import webbrowser
import os

def run_command(command):
    process = subprocess.Popen(command, shell=True)
    return process

def main():
    # Check if .env exists and has the API key
    if not os.path.exists(".env"):
        print("Error: .env file not found!")
        return
    
    with open(".env", "r") as f:
        if "GOOGLE_API_KEY" not in f.read():
            print("Error: GOOGLE_API_KEY not found in .env file!")
            return
    
    print("Starting FastAPI backend...")
    backend = run_command("start cmd /k python -m uvicorn backend:app --reload")
    
    # Wait for backend to start
    print("Waiting for backend to start...")
    time.sleep(3)
    
    print("Starting Streamlit frontend...")
    frontend = run_command("start cmd /k streamlit run app.py")
    
    # Open web browser
    webbrowser.open("http://localhost:8501")
    
    print("Services started!")
    print("Press Ctrl+C to exit...")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")
        # The processes will be terminated when their command windows are closed

if __name__ == "__main__":
    main()
