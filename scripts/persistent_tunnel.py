"""
Resilient Auto-Reconnecting Public Tunnel Service.
Maintains a standing public HTTPS tunnel for the local traffic sign server,
auto-reconnecting immediately upon any remote connection drop.
"""

import time
import subprocess
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
URL_FILE = BASE_DIR / "artifacts" / "live_public_url.txt"
URL_FILE.parent.mkdir(parents=True, exist_ok=True)

def run_persistent_tunnel():
    print("[*] Starting resilient public tunnel service for port 8000...")
    
    while True:
        try:
            print("[*] Initiating SSH connection to serveo.net...")
            cmd = [
                "ssh",
                "-o", "StrictHostKeyChecking=no",
                "-o", "ServerAliveInterval=15",
                "-o", "ServerAliveCountMax=3",
                "-R", "80:127.0.0.1:8000",
                "serveo.net"
            ]
            
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            for line in iter(proc.stdout.readline, ''):
                print(f"[Serveo] {line.strip()}")
                m = re.search(r'https://[a-zA-Z0-9\-\.]+\.serveousercontent\.com', line)
                if m:
                    live_url = m.group(0)
                    print("=" * 60)
                    print(f"🔥 ACTIVE PUBLIC HTTPS URL: {live_url}")
                    print("=" * 60)
                    with open(URL_FILE, "w", encoding="utf-8") as f:
                        f.write(live_url.strip())
                        
            proc.wait()
            print("[!] Tunnel disconnected. Reconnecting in 3 seconds...")
            time.sleep(3)
            
        except Exception as e:
            print(f"[!] Tunnel loop error: {e}. Reconnecting in 5 seconds...")
            time.sleep(5)

if __name__ == "__main__":
    run_persistent_tunnel()
