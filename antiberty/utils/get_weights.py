"""
Locate (and, if needed, download) the pre-trained AntiBERTy weights.

Resolution order:

1. an explicit path passed by the caller,
2. the ``ANTIBERTY_WEIGHTS_DIR`` environment variable,
3. a ``trained_models/AntiBERTy_md_smooth`` directory inside the installed package (e.g. a manual
   copy for offline machines),
4. the Hugging Face Hub repository ``jeffruffolo/AntiBERTy`` at a pinned revision, downloaded into
   the Hugging Face cache (``~/.cache/huggingface`` or ``$HF_HOME``).
"""

import os

import antiberty
from antiberty.utils.general import exists

HF_REPO_ID = "jeffruffolo/AntiBERTy"
HF_REVISION = "d582006e0c7dc6917d26fa60740a357e8df7b466"
WEIGHTS_DIR_ENV = "ANTIBERTY_WEIGHTS_DIR"
WEIGHT_FILES = ("config.json", "model.safetensors")

PACKAGED_WEIGHTS_DIR = os.path.join(
    os.path.dirname(os.path.realpath(antiberty.__file__)), "trained_models", "AntiBERTy_md_smooth"
)

_HINT = (
    f"Download config.json and model.safetensors from https://huggingface.co/{HF_REPO_ID} once and set "
    f"{WEIGHTS_DIR_ENV} to that directory, or pass checkpoint_path to AntiBERTyRunner."
)


def _is_weights_dir(path) -> bool:
    return (
        exists(path)
        and os.path.isfile(os.path.join(path, "config.json"))
        and (
            os.path.isfile(os.path.join(path, "model.safetensors"))
            or os.path.isfile(os.path.join(path, "pytorch_model.bin"))
        )
    )


def get_weights(path=None, download: bool = True) -> str:
    """
    Return a directory containing ``config.json`` and the model weights (see module docstring for
    the resolution order). ``download=False`` never touches the network.
    """
    if exists(path):
        if not _is_weights_dir(path):
            raise FileNotFoundError(f"{path} does not contain AntiBERTy weights (config.json + model.safetensors).")
        return path

    env_dir = os.environ.get(WEIGHTS_DIR_ENV)
    if exists(env_dir) and len(env_dir) > 0:
        if not _is_weights_dir(env_dir):
            raise FileNotFoundError(f"{WEIGHTS_DIR_ENV}={env_dir} does not contain AntiBERTy weights.")
        return env_dir

    if _is_weights_dir(PACKAGED_WEIGHTS_DIR):
        return PACKAGED_WEIGHTS_DIR

    if not download:
        raise FileNotFoundError(f"AntiBERTy weights not found. {_HINT}")

    return download_weights()


def download_weights(repo_id: str = HF_REPO_ID, revision=HF_REVISION) -> str:
    """Download (or fetch from cache) the weights from the Hugging Face Hub and return the local directory."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError, OfflineModeIsEnabled

    try:
        return snapshot_download(repo_id, revision=revision, allow_patterns=list(WEIGHT_FILES))
    except (LocalEntryNotFoundError, OfflineModeIsEnabled, HfHubHTTPError) as e:
        raise FileNotFoundError(
            f"Could not fetch AntiBERTy weights from https://huggingface.co/{repo_id} ({type(e).__name__}) "
            f"and no cached copy exists. {_HINT}"
        ) from e
