# src/data_prep.py
import os
from huggingface_hub import snapshot_download
from src.config import cfg


def download_snapshot():
    out = cfg.dataset_local_dir
    if not os.path.exists(out) or len(os.listdir(out)) == 0:
        print('Downloading dataset snapshot to', out)
        snapshot_download(repo_id=cfg.dataset_repo_id, repo_type='dataset', local_dir=out)
        print('Downloaded.')
    else:
        print('Snapshot already present at', out)


if __name__ == '__main__':
    download_snapshot()