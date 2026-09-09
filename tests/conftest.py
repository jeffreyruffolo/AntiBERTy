import os

import pytest

from antiberty.utils.get_weights import get_weights

HEAVY = "EVQLVQSGPEVKKPGTSVKVSCKASGFTFMSSAVQWVRQARGQRLEWIGWIVIGSGNTNYAQKFQERVTITRDMSTSTAYMELSSLRSEDTAVYYCAAPYCSSISCNDGFDIWGQGTMVTVS"
LIGHT = (
    "DVVMTQTPFSLPVSLGDQASISCRSSQSLVHSNGNTYLHWYLQKPGQSPKLLIYKVSNRFSGVPDRFSGSGSGTDFTLKISRVEAEDLGVYFCSQSTHVPYTFGGGTKLEIK"
)


def _weights_available() -> bool:
    try:
        get_weights()
        return True
    except (FileNotFoundError, OSError):
        if os.environ.get("ANTIBERTY_REQUIRE_WEIGHTS"):
            raise
        return False


requires_weights = pytest.mark.skipif(not _weights_available(), reason="pre-trained AntiBERTy weights not available")


@pytest.fixture(scope="session")
def runner():
    from antiberty import AntiBERTyRunner

    return AntiBERTyRunner(device="cpu")
