import subprocess

def get_last_commit_hash():
    try:
        commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('utf-8')
        return commit_hash
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        return None

"""
# Example usage:
last_commit = get_last_commit_hash()
if last_commit:
    print(f"Last commit hash: {last_commit}")
"""