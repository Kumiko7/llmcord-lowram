import os
import sys
import time
import zipfile
import yaml
import requests

# --- Configuration ---
CONFIG_FILE = 'config.yaml'
ZIP_FILENAME = 'upload.zip'
DISCLOUD_API_BASE_URL = "https://api.discloud.app/v2"
EXCLUDE_DIRS = {'.git'}
EXCLUDE_FILES = {'.gitignore', 'deploy.bat', 'deploy_helper.py', ZIP_FILENAME}

def cleanup_and_exit(error_message, exit_code=1):
    """Prints an error, cleans up the zip file, and exits."""
    print(f"\n[ERROR] {error_message}")
    if os.path.exists(ZIP_FILENAME):
        try:
            os.remove(ZIP_FILENAME)
            print(f"Cleaned up {ZIP_FILENAME}.")
        except OSError as e:
            print(f"Error while cleaning up {ZIP_FILENAME}: {e}")
    sys.exit(exit_code)

def main():
    # --- 1. Load Config ---
    print("Reading configuration from config.yaml...")
    try:
        with open(CONFIG_FILE, 'r', encoding="utf-8") as f:
            config = yaml.safe_load(f)
        app_id = config.get('discloud_app')
        api_token = config.get('discloud_token')
        if not app_id or not api_token:
            cleanup_and_exit("'discloud_app' or 'discloud_token' not found in config.yaml.")
    except FileNotFoundError:
        cleanup_and_exit(f"{CONFIG_FILE} not found in the current directory.")
    except Exception as e:
        cleanup_and_exit(f"Failed to read or parse {CONFIG_FILE}: {e}")
    
    print(f"Found Discloud App ID: {app_id}")

    # --- 2. Create Zip File with Exclusions ---
    print(f"Creating {ZIP_FILENAME} for deployment...")
    try:
        with zipfile.ZipFile(ZIP_FILENAME, 'w', zipfile.ZIP_DEFLATED) as zf:
            for root, dirs, files in os.walk('.'):
                # Modify dirs in-place to prevent walking into excluded directories
                dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
                
                for file in files:
                    if file in EXCLUDE_FILES:
                        continue
                    file_path = os.path.join(root, file)
                    # The second argument to write() is the path inside the zip
                    zf.write(file_path, os.path.relpath(file_path, '.'))
        print(f"Successfully created {ZIP_FILENAME}.")
    except Exception as e:
        cleanup_and_exit(f"Failed to create zip file: {e}")

    # --- 3. Post to Discloud Commit API ---
    commit_url = f"{DISCLOUD_API_BASE_URL}/app/{app_id}/commit"
    headers = {'api-token': str(api_token)}
    
    print(f"Uploading to Discloud...")
    try:
        with open(ZIP_FILENAME, 'rb') as f:
            files = {'file': f}
            response = requests.put(commit_url, headers=headers, files=files, timeout=60)
        
        response.raise_for_status() # Raises an exception for 4xx/5xx status codes
        print(f"Upload successful! Discloud says: {response.json().get('message')}")

    except requests.exceptions.RequestException as e:
        error_details = e.response.text if e.response else "No response from server."
        cleanup_and_exit(f"API request to Discloud failed: {e}\nDetails: {error_details}")

    # --- 4. Wait for App to Restart ---
    wait_time = 20
    print(f"\nWaiting {wait_time} seconds for the app to restart...")
    time.sleep(wait_time)

    # --- 5. Get App Status ---
    status_url = f"{DISCLOUD_API_BASE_URL}/app/{app_id}/status"
    print("Checking app status...")
    try:
        response = requests.get(status_url, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()

        if data.get('status') == 'ok' and data['apps']['container'] == 'Online':
            print("\n[SUCCESS] App status is 'Online'.")
            print(f"Memory: {data['apps']['memory']}")
            print(f"Last Restart: {data['apps']['last_restart']}")
        else:
            status = data.get('apps', {}).get('container', 'Unknown')
            cleanup_and_exit(f"App is not online. Current status: '{status}'")

    except requests.exceptions.RequestException as e:
        cleanup_and_exit(f"Failed to get app status: {e}")

    # --- Final Cleanup and Successful Exit ---
    if os.path.exists(ZIP_FILENAME):
        os.remove(ZIP_FILENAME)
    
    # Exit with code 0 to signal success to the .bat file
    sys.exit(0)

if __name__ == "__main__":
    main()