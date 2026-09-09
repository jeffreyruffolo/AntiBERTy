import pytest
import torch
from conftest import HEAVY, LIGHT, requires_weights

pytestmark = requires_weights


def test_device_argument(runner):
    assert runner.device == torch.device("cpu")
    assert next(runner.model.parameters()).device == torch.device("cpu")


def test_embed_shapes(runner):
    emb, att = runner.embed([HEAVY, LIGHT], return_attention=True)
    assert emb[0].shape == (len(HEAVY) + 2, 512)
    assert emb[1].shape == (len(LIGHT) + 2, 512)
    assert att[0].shape == (8, 8, len(HEAVY) + 2, len(HEAVY) + 2)
    assert torch.isfinite(emb[0]).all()


def test_embed_all_layers_and_single_string(runner):
    emb = runner.embed([HEAVY], hidden_layer=None)
    assert emb[0].shape == (9, len(HEAVY) + 2, 512)
    assert torch.allclose(runner.embed(HEAVY)[0], emb[0][-1])


def test_classify(runner):
    species, chains = runner.classify([HEAVY, LIGHT])
    assert chains == ["Heavy", "Light"]
    assert species[0] == "Human"


TRASTUZUMAB_VH = "QVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS"


def test_graft_prediction(runner):
    species, chains, grafts = runner.classify([HEAVY, TRASTUZUMAB_VH], return_graft=True)
    assert grafts == ["Natural", "Grafted"] and chains == ["Heavy", "Heavy"]
    p = runner.graft_probability([HEAVY, TRASTUZUMAB_VH])
    assert p.shape == (2,) and p[0] < 0.1 and p[1] > 0.9


def test_classify_with_labels(runner):
    scores = runner.classify(HEAVY, species_label="Human", chain_label="Heavy")
    assert set(scores) == {"species_ll", "chain_ll"} and scores["chain_ll"] <= 0
    with pytest.raises(ValueError, match="one sequence"):
        runner.classify([HEAVY, LIGHT], species_label="Human")


def test_fill_masks(runner):
    masked = HEAVY[:10] + "___" + HEAVY[13:]
    filled = runner.fill_masks([masked])[0]
    assert len(filled) == len(HEAVY)
    assert "_" not in filled and filled[:10] == HEAVY[:10] and filled[13:] == HEAVY[13:]


def test_fill_masks_keeps_unknown_residues(runner):
    filled = runner.fill_masks("EVXQL__SG")[0]
    assert len(filled) == 9 and filled[2] == "X" and "_" not in filled


def test_pseudo_log_likelihood(runner):
    pll = runner.pseudo_log_likelihood([HEAVY[:40]])
    assert pll.shape == (1,)
    assert torch.isfinite(pll).all() and pll.item() < 0
    chunked = runner.pseudo_log_likelihood([HEAVY[:40]], batch_size=7)
    assert torch.allclose(pll, chunked, atol=1e-5)


def test_pseudo_log_likelihood_with_nonstandard_residues(runner):
    pll = runner.pseudo_log_likelihood(["EVQL_SGXKLVQSGP"])
    assert torch.isfinite(pll).all()
    with pytest.raises(ValueError, match="no standard residues"):
        runner.pseudo_log_likelihood(["___"])
