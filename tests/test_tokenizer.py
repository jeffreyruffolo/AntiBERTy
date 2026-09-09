import pytest
import torch

from antiberty.tokenizer import VOCAB, AntiBERTyTokenizer


@pytest.fixture(scope="module")
def tok():
    return AntiBERTyTokenizer()


def test_vocab(tok):
    assert len(tok) == 25 and tok.vocab == VOCAB
    assert tok.all_special_ids == [0, 1, 2, 3, 4]


def test_vocab_file(tmp_path):
    path = tmp_path / "vocab.txt"
    path.write_text("\n".join(VOCAB) + "\n")
    assert AntiBERTyTokenizer(str(path)).vocab == VOCAB


def test_encode_adds_cls_sep(tok):
    ids = tok.encode("EVQL")
    assert ids[0] == tok.cls_token_id and ids[-1] == tok.sep_token_id
    assert tok.convert_ids_to_tokens(ids) == ["[CLS]", "E", "V", "Q", "L", "[SEP]"]


def test_mask_unk_case_and_whitespace(tok):
    assert tok.tokenize("E_x") == ["E", "[MASK]", "[UNK]"]
    assert tok.encode("evql") == tok.encode("EVQL") == tok.encode(" EVQL\n")
    assert tok.tokenize("E V [MASK] L") == ["E", "V", "[MASK]", "L"]
    assert tok.tokenize(["E", "[MASK]", "L"]) == ["E", "[MASK]", "L"]
    with pytest.raises(ValueError, match="masked residues"):
        tok.tokenize("EV[MASK]L")
    with pytest.raises(ValueError, match="empty"):
        tok.encode("")


def test_batch_padding(tok):
    out = tok(["EVQL", "EV"])
    assert out["input_ids"].shape == (2, 6)
    assert out["attention_mask"].tolist() == [[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]]
    assert out["input_ids"][1, -1].item() == tok.pad_token_id
    assert torch.equal(tok("EVQL")["input_ids"], tok(["EVQL"])["input_ids"])


def test_decode_roundtrip(tok):
    seq = "QVQLQESGGGLVQAGGSLTLSCAVSG"
    assert tok.decode(tok.encode(seq)) == seq
    assert tok.batch_decode(tok([seq, "EV"])["input_ids"]) == [seq, "EV"]


def test_too_long(tok):
    with pytest.raises(ValueError):
        tok.encode("A" * 511)
