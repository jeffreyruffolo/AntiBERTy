import pytest
from conftest import HEAVY, LIGHT, requires_weights

from antiberty.cli import _read_sequences, build_parser, main, read_fasta


def test_parser():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])
    args = build_parser().parse_args(["embed", "EVQL", "-o", "x.pt", "--pool", "mean"])
    assert args.pool == "mean" and args.sequences == ["EVQL"]


def test_requires_sequences():
    with pytest.raises(SystemExit, match="provide sequences"):
        main(["classify"])


def test_read_fasta_and_unique_ids(tmp_path):
    fasta = tmp_path / "a.fasta"
    fasta.write_text(">H description here\nEVQL\nVQSG\n>H\nDVVM\n>L\nDIQM\n")
    assert list(read_fasta(str(fasta))) == [("H", "EVQLVQSG"), ("H", "DVVM"), ("L", "DIQM")]
    ids, seqs = _read_sequences(build_parser().parse_args(["classify", "--fasta", str(fasta)]))
    assert ids == ["H", "H_2", "L"] and seqs[0] == "EVQLVQSG"


@requires_weights
def test_embed_and_classify(tmp_path, capsys):
    import torch

    out = tmp_path / "emb.pt"
    assert main(["embed", HEAVY, LIGHT, "-o", str(out), "--pool", "mean", "--device", "cpu"]) == 0
    saved = torch.load(str(out), weights_only=True)
    assert set(saved) == {"seq1", "seq2"} and saved["seq1"].shape == (512,)

    npz = tmp_path / "emb.npz"
    assert main(["embed", HEAVY, "-o", str(npz), "--pool", "residue", "--attention", "--device", "cpu"]) == 0
    import numpy as np

    data = np.load(str(npz))
    assert data["seq1"].shape == (len(HEAVY), 512) and data["seq1_attention"].shape[-1] == len(HEAVY) + 2

    fasta = tmp_path / "in.fasta"
    fasta.write_text(f">H\n{HEAVY}\n>L\n{LIGHT}\n")
    assert main(["classify", "--fasta", str(fasta), "--device", "cpu"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert (
        lines[0] == "id\tspecies\tchain"
        and lines[1].split("\t") == ["H", "Human", "Heavy"]
        and lines[2].endswith("Light")
    )

    assert main(["classify", HEAVY, "--graft", "--device", "cpu"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[0].split("\t") == ["id", "species", "chain", "graft", "p_graft"]
    assert lines[1].split("\t")[3] == "Natural"


@requires_weights
def test_fill_and_pll(capsys):
    masked = HEAVY[:10] + "___" + HEAVY[13:]
    assert main(["fill", masked, "--device", "cpu"]) == 0
    filled = capsys.readouterr().out.strip()
    assert len(filled) == len(HEAVY) and "_" not in filled

    assert main(["pll", HEAVY[:40], "--device", "cpu"]) == 0
    out = capsys.readouterr().out.strip().splitlines()
    assert out[0] == "id\tpseudo_log_likelihood" and float(out[1].split("\t")[1]) < 0
