from huggingface_hub import snapshot_download
import requests
while True:
    try:
        snapshot_download(repo_id="OpenGVLab/InternVL-Chat-V1-5", local_dir="../OpenGVLab/InternVL-Chat-V1-5", local_dir_use_symlinks=False, max_workers=1, resume_download=True)
        break
    except requests.exceptions.ConnectionError as e:
        print(f"{e}")