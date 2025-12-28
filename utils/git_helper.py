import subprocess
import sys

def push_changes(message="Update from Colab"):
    """
    Commits and pushes changes using standard Python subprocesses.
    Valid for use in .py files.
    """
    print(f"📤 Preparing to push changes: '{message}'")
    
    try:
        # 1. Configure Identity
        subprocess.run(["git", "config", "--global", "user.email", "chandan.kr.singh.11@gmail.com"], check=True)
        subprocess.run(["git", "config", "--global", "user.name", "chandan110791"], check=True)
        
        # 2. Add All Changes
        # We capture output to avoid spamming logs unless there's an error
        subprocess.run(["git", "add", "."], check=True)
        
        # 3. Commit
        # We allow this to fail if there's nothing to commit (e.g., no changes)
        result = subprocess.run(["git", "commit", "-m", message], capture_output=True, text=True)
        if result.returncode != 0:
            if "nothing to commit" in result.stdout:
                print("⚠️ No changes to commit.")
                return
            else:
                print(f"❌ Commit failed: {result.stderr}")
                return

        # 4. Push
        subprocess.run(["git", "push"], check=True)
        print("✅ Changes successfully pushed to GitHub.")

    except subprocess.CalledProcessError as e:
        print(f"❌ Git operation failed: {e}")