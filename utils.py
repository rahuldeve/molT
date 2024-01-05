import sys

import wandb


def download_model_from_wandb(entity, project, run_name, alias):
    artifact_str = f"{entity}/{project}/model-{run_name}:{alias}"
    api = wandb.Api()
    artifact = api.artifact(artifact_str)
    artifact.download()

    if sys.platform.startswith("win32"):
        model_dir = f"artifacts/model-{run_name}-{alias}"
    else:
        model_dir = f"artifacts/model-{run_name}:{alias}"

    return model_dir
