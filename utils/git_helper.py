def push_changes(message="Update from Colab"):
    """
    Commits and pushes all changes in the current directory to GitHub.
    """
    import os
    from google.colab import userdata
    
    print(f"📤 Pushing changes: '{message}'")
    
    # Configure Identity (Required for CI/CD)
    !git config --global user.email "chandan.kr.singh.11@gmail.com"
    !git config --global user.name "Chandan Singh"
    
    # Add, Commit, Push
    !git add .
    !git commit -m "{message}"
    !git push
    
    print("✅ Changes saved to GitHub.")