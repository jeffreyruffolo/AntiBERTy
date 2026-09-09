"""
Command-line interface for AntiBERTy: ``antiberty embed | classify | fill | pll``.
"""

import argparse
import sys
from importlib.metadata import PackageNotFoundError, version

from antiberty.tokenizer import MASK_CHAR


def read_fasta(path):
    """Yield (id, sequence) pairs from a FASTA file; the id is the first word of the header."""
    header, chunks = None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks)
                header, chunks = line[1:].split()[0] if len(line) > 1 else "", []
            elif line:
                chunks.append(line)
    if header is not None:
        yield header, "".join(chunks)


def _read_sequences(args):
    """Sequences and unique ids from positional arguments and/or a FASTA file."""
    ids, seqs = [], []
    for i, s in enumerate(args.sequences):
        ids.append(f"seq{i + 1}")
        seqs.append(s)
    if args.fasta:
        for rid, seq in read_fasta(args.fasta):
            ids.append(rid)
            seqs.append(seq)
    if len(seqs) == 0:
        raise SystemExit("error: provide sequences as arguments or with --fasta")

    seen = {}
    for i, rid in enumerate(ids):
        seen[rid] = seen.get(rid, 0) + 1
        if seen[rid] > 1:
            ids[i] = f"{rid}_{seen[rid]}"

    return ids, seqs


def _add_common(p, masks=False):
    p.add_argument(
        "sequences",
        nargs="*",
        metavar="SEQ",
        help="Sequence(s)" + (f"; use '{MASK_CHAR}' for masked residues." if masks else "."),
    )
    p.add_argument("--fasta", metavar="FILE", help="FASTA file of sequences (record ids are kept in the output).")
    p.add_argument("--device", help='Torch device, e.g. "cpu", "cuda:1", "mps" (default: CUDA if available).')


def _runner(args):
    from antiberty import AntiBERTyRunner

    return AntiBERTyRunner(device=args.device)


def _run_embed(args):
    import torch

    ids, seqs = _read_sequences(args)
    layer = None if args.layer == "all" else int(args.layer)
    if layer is not None and not -9 <= layer <= 8:
        raise SystemExit("error: --layer must be 0-8 (or -1 for the last layer, or 'all')")
    runner = _runner(args)
    out = runner.embed(seqs, hidden_layer=layer, return_attention=args.attention)
    embs, atts = out if args.attention else (out, None)

    result = {}
    for i, (sid, e) in enumerate(zip(ids, embs)):
        e = e.detach().cpu()
        if args.pool == "mean":
            e = e[..., 1:-1, :].mean(dim=-2)  # drop [CLS]/[SEP] before pooling
        elif args.pool == "residue":
            e = e[..., 1:-1, :]
        result[sid] = e
        if atts is not None:
            result[f"{sid}_attention"] = atts[i].detach().cpu()

    if args.output.endswith(".npz"):
        import numpy as np

        np.savez(args.output, **{k: v.numpy() for k, v in result.items()})
    else:
        torch.save(result, args.output)
    print(f"Wrote {len(ids)} embedding(s) to {args.output}", file=sys.stderr)

    return 0


def _run_classify(args):
    ids, seqs = _read_sequences(args)
    runner = _runner(args)
    species, chains, grafts = runner.classify(seqs, return_graft=True)
    if args.graft:
        p_graft = runner.graft_probability(seqs).tolist()
        print("id\tspecies\tchain\tgraft\tp_graft")
        for sid, sp, ch, g, p in zip(ids, species, chains, grafts, p_graft):
            print(f"{sid}\t{sp}\t{ch}\t{g}\t{p:.3f}")
    else:
        print("id\tspecies\tchain")
        for sid, sp, ch in zip(ids, species, chains):
            print(f"{sid}\t{sp}\t{ch}")

    return 0


def _run_fill(args):
    ids, seqs = _read_sequences(args)
    filled = _runner(args).fill_masks(seqs)
    for sid, f in zip(ids, filled):
        print(f">{sid}\n{f}" if args.fasta_out else f)

    return 0


def _run_pll(args):
    ids, seqs = _read_sequences(args)
    pll = _runner(args).pseudo_log_likelihood(seqs, batch_size=args.batch_size, verbose=len(seqs) > 1)
    print("id\tpseudo_log_likelihood")
    for sid, v in zip(ids, pll.tolist()):
        print(f"{sid}\t{v:.4f}")

    return 0


def build_parser():
    parser = argparse.ArgumentParser(prog="antiberty", description="AntiBERTy antibody language model.")
    try:
        pkg_version = version("antiberty")
    except PackageNotFoundError:
        pkg_version = "unknown"
    parser.add_argument("--version", action="version", version=f"antiberty {pkg_version}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("embed", help="Write per-residue or pooled embeddings to a .pt or .npz file.")
    _add_common(p, masks=True)
    p.add_argument("-o", "--output", required=True, metavar="FILE", help="Output file: .pt (torch.save dict) or .npz.")
    p.add_argument(
        "--layer", default="-1", help="Hidden layer to use, 0-8, or 'all' for every layer (default: -1, the last)."
    )
    p.add_argument(
        "--pool",
        choices=["none", "residue", "mean"],
        default="none",
        help="none: (L+2)x512 with [CLS]/[SEP]; residue: Lx512; mean: 512-d mean over residues.",
    )
    p.add_argument(
        "--attention", action="store_true", help="Also save attention matrices (layers x heads x (L+2) x (L+2))."
    )
    p.set_defaults(func=_run_embed)

    p = sub.add_parser("classify", help="Predict species and chain type (TSV to stdout).")
    _add_common(p, masks=True)
    p.add_argument(
        "--graft",
        action="store_true",
        help="Also report whether the sequence looks CDR-grafted (humanized), with probability.",
    )
    p.set_defaults(func=_run_classify)

    p = sub.add_parser("fill", help=f"Fill masked residues ('{MASK_CHAR}') with the most likely amino acid.")
    _add_common(p, masks=True)
    p.add_argument("--fasta-out", action="store_true", help="Print FASTA records instead of bare sequences.")
    p.set_defaults(func=_run_fill)

    p = sub.add_parser("pll", help="Pseudo log-likelihood per sequence (TSV to stdout).")
    _add_common(p)
    p.add_argument("--batch-size", type=int, default=64, help="Masked copies scored per forward pass (default 64).")
    p.set_defaults(func=_run_pll)

    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
