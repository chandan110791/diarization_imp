import subprocess
import os
import sys

def push_changes(message="Update from Colab"):
    """
    Commits and pushes changes. 
    Auto-authenticates if running on Google Colab.
    """
    print(f"📤 Preparing to push: '{message}'")
    
    # --- COLAB AUTHENTICATION BLOCK (CRITICAL) ---
    # Only runs if we are in Colab
    if os.path.exists("/content"):
        try:
            from google.colab import userdata
            token = userdata.get('GITHUB_TOKEN')
            
            # HARDCODED REPO DETAILS (Update these if they change)
            USERNAME = "chandan110791"
            REPO = "diarization_imp"
            
            # Construct the authenticated URL
            auth_url = f"https://{token}@github.com/{USERNAME}/{REPO}.git"
            
            # 1. Update Remote URL to include Token
            subprocess.run(["git", "remote", "set-url", "origin", auth_url], check=True)
            
            # 2. Configure Identity
            subprocess.run(["git", "config", "--global", "user.email", "chandan.kr.singh.11@gmail.com"], check=True)
            subprocess.run(["git", "config", "--global", "user.name", "chandan110791"], check=True)
            
        except Exception as e:
            print(f"⚠️ Auth Warning: Could not auto-configure Colab token. ({e})")
            print("   (Ensure 'GITHUB_TOKEN' is in your Secrets with Notebook Access ON)")

    # --- STANDARD GIT COMMANDS ---
    try:
        # 3. Add
        subprocess.run(["git", "add", "."], check=True)
        
        # 4. Commit
        result = subprocess.run(["git", "commit", "-m", message], capture_output=True, text=True)
        
        # Handle "nothing to commit" gracefully
        if result.returncode != 0:
            if "nothing to commit" in result.stdout:
                print("⚠️ No changes to commit.")
                # We proceed to push anyway just in case local commits exist that haven't been pushed
            else:
                print(f"❌ Commit failed: {result.stderr}")
                return
        
        # 5. Push
        subprocess.run(["git", "push"], check=True)
        print("✅ Success! Changes pushed to GitHub.")

    except subprocess.CalledProcessError as e:
        print(f"❌ Git Command Failed: {e}")