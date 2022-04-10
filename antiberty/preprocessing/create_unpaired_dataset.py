import os
import argparse
from glob import glob
import datasets
import transformers

import deeph3


def csv_to_dataset(csv_files,
                   out_dir,
                   num_proc,
                   cache_dir,
                   percent=0.2,
                   save_test=False,
                   keep_cdrs=False):
    tokenizer = transformers.BertTokenizerFast(
        vocab_file="deeph3/models/AntiBERTy/vocab.txt", do_lower_case=False)

    def tokenize_seqs(example):
        seq = list(example["seq"])
        example["seq"] = tokenizer(" ".join(seq))["input_ids"]

        return example

    dataset = datasets.load_dataset("csv",
                                    data_files=csv_files,
                                    cache_dir=cache_dir)["train"]

    if keep_cdrs:
        dataset = dataset.remove_columns(
            ["isotype", "b_type", "b_source", "disease", "vaccine"])
    else:
        dataset = dataset.remove_columns([
            "cdr1", "cdr2", "cdr3", "isotype", "b_type", "b_source", "disease",
            "vaccine"
        ])

    if percent < 1:
        dataset = dataset.train_test_split(test_size=(1 - percent))
        train_dataset = dataset["train"]
        test_dataset = dataset["test"]

    train_dataset = train_dataset.map(tokenize_seqs, num_proc=num_proc)
    train_dataset = train_dataset.rename_column("seq", "input_ids")

    if save_test:
        train_dataset.save_to_disk(os.path.join(out_dir, "train"))

        test_dataset = test_dataset.map(tokenize_seqs, num_proc=num_proc)
        test_dataset = test_dataset.rename_column("seq", "input_ids")
        test_dataset.save_to_disk(os.path.join(out_dir, "test"))
    else:
        train_dataset.save_to_disk(out_dir)


def cli():
    project_path = os.path.abspath(os.path.join(deeph3.__file__, "../.."))
    data_path = os.path.join(project_path, "data")

    desc = 'Creates Huggingface dataset from OAS unpaired csv files'
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
    parser.add_argument(
        '--num_proc',
        type=int,
        help='Number of processes to use for dataset generation',
        default=12)
    parser.add_argument(
        '--percent',
        type=float,
        help='Number of processes to use for dataset generation',
        default=0.2)
    parser.add_argument(
        '--cache_dir',
        type=str,
        help='Number of processes to use for dataset generation',
        default="~/datasets")
    parser.add_argument('--save_test', default=False, action='store_true')
    parser.add_argument('--keep_cdrs', default=False, action='store_true')

    args = parser.parse_args()
    oas_csv_dir = args.oas_csv_dir
    out_dir = args.out_dir
    num_proc = args.num_proc
    percent = args.percent
    cache_dir = args.cache_dir
    save_test = args.save_test
    keep_cdrs = args.keep_cdrs

    if not args.overwrite and os.path.exists(out_dir):
        exit("Dataset already exists: {}".format(out_dir))

    csv_files = list(glob(os.path.join(oas_csv_dir, "*.csv")))
    if oas_csv_dir[-4:] == ".csv" and os.path.exists(oas_csv_dir):
        csv_files = [oas_csv_dir]

    csv_to_dataset(csv_files,
                   out_dir,
                   num_proc=num_proc,
                   cache_dir=cache_dir,
                   percent=percent,
                   save_test=save_test,
                   keep_cdrs=keep_cdrs)


if __name__ == '__main__':
    cli()
