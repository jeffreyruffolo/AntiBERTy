import os
import argparse
from glob import glob
import datasets
import transformers

import deeph3


def csv_to_dataset(csv_files, out_dir):
    tokenizer = transformers.BertTokenizerFast(
        vocab_file="deeph3/models/AntiBERTy/vocab.txt", do_lower_case=False)

    def combine_hl_seqs(example):
        seq = list(example["hseq"]) + list(example["lseq"])
        example["hseq"] = tokenizer(" ".join(seq))["input_ids"]

        return example

    dataset = datasets.load_dataset("csv", data_files=csv_files)["train"]

    dataset = dataset.remove_columns([
        "h1", "h2", "h3", "l1", "l2", "l3", "isotype", "b_type", "b_source",
        "disease", "vaccine"
    ])

    dataset = dataset.map(combine_hl_seqs)
    dataset = dataset.remove_columns(["lseq"])
    dataset = dataset.rename_column("hseq", "input_ids")

    dataset.save_to_disk(out_dir)


def cli():
    project_path = os.path.abspath(os.path.join(deeph3.__file__, "../.."))
    data_path = os.path.join(project_path, "data")

    desc = 'Creates Huggingface dataset from OAS paired csv files'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        'oas_csv_dir',
        type=str,
        help='The directory containing processed antibody sequence CSV files')
    parser.add_argument(
        'out_dir',
        type=str,
        help='The path to store the outputted Huggingface dataset')
    parser.add_argument('--overwrite',
                        type=bool,
                        help='Whether or not to overwrite a file or not,'
                        ' if it exists',
                        default=False)

    args = parser.parse_args()
    oas_csv_dir = args.oas_csv_dir
    out_dir = args.out_dir

    if not args.overwrite and os.path.exists(out_dir):
        exit("Dataset already exists: {}".format(out_dir))

    csv_files = list(glob(os.path.join(oas_csv_dir, "*.csv")))
    csv_to_dataset(csv_files, out_dir)


if __name__ == '__main__':
    cli()
