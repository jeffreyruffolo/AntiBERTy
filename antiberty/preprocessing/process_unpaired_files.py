import multiprocessing
import joblib
import h5py
import random
import numpy as np
import argparse
from numpy.core.getlimits import _register_known_types
from tqdm import tqdm
import os
from glob import glob
from pathlib import Path
import pandas as pd
import json
import itertools

import deeph3
from deeph3.util.util import _aa_dict, letter_to_num

cdr_names = ["cdr1", "cdr2", "cdr3"]
seq_components = {
    "heavy": ["fwh1", "cdrh1", "fwh2", "cdrh2", "fwh3", "cdrh3", "fwh4"],
    "light": ["fwl1", "cdrl1", "fwl2", "cdrl2", "fwl3", "cdrl3", "fwl4"]
}


def to_dict(text):
    try:
        return json.loads(text.replace("\\\'", "\\\"").replace("\'", "\""))
    except json.decoder.JSONDecodeError:
        return json.loads(text.replace("\\\'", "\\\""))


def combine_anarci_components(anarci_dict, components):
    component_seqs = {}
    for c in components:
        component_list = [(k, v) for k, v in anarci_dict[c].items()]
        sorted_component_list = list(
            sorted(
                component_list,
                key=lambda x:
                (x[0] + "{}".format(""
                                    if x[0][-1].isalpha() else "0")).zfill(5)))
        component_seq = "".join([x[1] for x in sorted_component_list])
        component_seqs[c] = component_seq

    seq = "".join(component_seqs.values())

    return seq, component_seqs


def extract_seq_components(anarci_dict, seq_components, cdr_names):
    seq, component_seqs = combine_anarci_components(anarci_dict,
                                                    seq_components)

    component_ranges = [
        len("".join(list(component_seqs.values())[:j]))
        for j in range(len(seq_components))
    ]
    component_ranges = [(component_ranges[j], component_ranges[j + 1] - 1)
                        for j in range(len(component_ranges) - 1)]

    cdr_range_dict = {}
    cdr_seq_dict = {}
    for i in range(3):
        cdr_range_dict[cdr_names[i]] = component_ranges[2 * i + 1]
        cdr_seq_dict[cdr_names[i]] = list(component_seqs.values())[2 * i + 1]

    return seq, cdr_range_dict, cdr_seq_dict


def process_csv_data(csv_file,
                     out_file_template,
                     print_progress=True,
                     verbose=False):
    rep_info = to_dict(
        np.genfromtxt(csv_file,
                      max_rows=1,
                      dtype=str,
                      delimiter="\t",
                      comments=None).item())
    if rep_info["Size"] == 0:
        return

    info_dict = {
        "species": rep_info["Species"],
        "chain_type": rep_info["Chain"],
        "isotype": rep_info["Isotype"],
        "b_type": rep_info["BType"],
        "b_source": rep_info["BSource"],
        "disease": rep_info["Disease"],
        "vaccine": rep_info["Vaccine"],
    }
    info_arr = np.array(list(info_dict.values()))

    col_names = pd.read_csv(csv_file, skiprows=1, nrows=1).columns
    max_rows = int(1e6)
    reader = pd.read_csv(csv_file,
                         skiprows=2,
                         chunksize=max_rows,
                         names=col_names,
                         header=None,
                         usecols=["ANARCI_status", "ANARCI_numbering"])
    for chunk_i, df in enumerate(reader):
        if os.path.exists(out_file_template.format(chunk_i)):
            continue

        df = df[['ANARCI_numbering']]

        data_list = []
        missing_component = False
        for index, anarci_data in enumerate(df.values):
            anarci_data = anarci_data.item()
            # row_seq_components = seq_components[
            #     info_dict["chain_type"].lower()]
            row_seq_components = seq_components[
                "heavy"] if "cdrh1" in anarci_data else seq_components["light"]
            for c in row_seq_components:
                if not c in to_dict(anarci_data):
                    if verbose:
                        print("Missing heavy component in index {}: {}".format(
                            index, c))
                    missing_component = True

            if missing_component:
                continue

            # Extract sequence from OAS data
            seq, cdr_range_dict, cdr_seq_dict = extract_seq_components(
                to_dict(anarci_data), row_seq_components, cdr_names)

            seq_arr = np.concatenate([
                info_arr,
                np.array([
                    seq, *list(cdr_seq_dict.values()),
                    *[str(r) for r in cdr_range_dict.values()]
                ])
            ])

            data_list.append(seq_arr)

        data_arr = np.stack(data_list)
        np.savetxt(out_file_template.format(chunk_i),
                   data_arr,
                   delimiter="\t",
                   fmt="%s")


def process_json_data(json_file,
                      out_file_template,
                      print_progress=True,
                      verbose=False):
    rep_info = to_dict(
        np.genfromtxt(json_file,
                      max_rows=1,
                      dtype=str,
                      delimiter="\t",
                      comments=None).item())
    if rep_info["Size"] == 0:
        return

    info_dict = {
        "species": rep_info["Species"],
        "chain_type": rep_info["Chain"],
        "isotype": rep_info["Isotype"],
        "b_type": rep_info["BType"],
        "b_source": rep_info["BSource"],
        "disease": rep_info["Disease"],
        "vaccine": rep_info["Vaccine"],
    }
    info_arr = np.array(list(info_dict.values()))

    max_rows = int(1e6)
    reader = pd.read_csv(json_file,
                         skiprows=1,
                         header=None,
                         delimiter="\t",
                         chunksize=max_rows)
    for chunk_i, df in enumerate(reader):
        if os.path.exists(out_file_template.format(chunk_i)):
            continue

        data_list = []
        missing_component = False
        for index, row_data in enumerate(df.values):
            row_data = to_dict(row_data.item())
            anarci_data = row_data["data"]
            # row_seq_components = seq_components[
            #     info_dict["chain_type"].lower()]
            row_seq_components = seq_components[
                "heavy"] if "cdrh1" in anarci_data else seq_components["light"]
            for c in row_seq_components:
                if not c in to_dict(anarci_data):
                    if verbose:
                        print("Missing heavy component in index {}: {}".format(
                            index, c))
                    missing_component = True

            if missing_component:
                continue

            # Extract sequence from OAS data
            seq, cdr_range_dict, cdr_seq_dict = extract_seq_components(
                to_dict(anarci_data), row_seq_components, cdr_names)

            seq_arr = np.concatenate([
                info_arr,
                np.array([
                    seq, *list(cdr_seq_dict.values()),
                    *[str(r) for r in cdr_range_dict.values()]
                ])
            ])

            data_list.append(seq_arr)

        data_arr = np.stack(data_list)
        np.savetxt(out_file_template.format(chunk_i),
                   data_arr,
                   delimiter="\t",
                   fmt="%s")


def process_zipped_oas_files(oas_dir,
                             out_dir,
                             size_sort=True,
                             reproc_csv=False,
                             reproc_json=False,
                             verbose=False):
    if size_sort:
        zipped_files = list(
            sorted(glob(os.path.join(oas_dir, "*.gz")),
                   key=os.path.getsize,
                   reverse=True))
    else:
        zipped_files = list(glob(os.path.join(oas_dir, "*.gz")))
        random.shuffle(zipped_files)

    zipped_json_files = [zf for zf in zipped_files if ".json.gz" in zf]
    zipped_csv_files = [zf for zf in zipped_files if ".csv.gz" in zf]

    out_json_file_template = os.path.join(out_dir, "{}.proc{{}}.tsv")
    out_csv_file_template = os.path.join(out_dir, "{}.proc{{}}.tsv")

    if reproc_json:
        for zf in zipped_json_files:
            json_files = out_json_file_template.format(
                os.path.split(zf)[1][:-8]).format("*")
            os.system("rm -f {}".format(json_files))
    if verbose:
        print("Total zipped JSON files:\t", len(zipped_json_files))
    zipped_json_files = [
        zf for zf in zipped_json_files if not os.path.exists(
            out_json_file_template.format(os.path.split(zf)[1][:-8]).format(0))
    ]
    if verbose:
        print("Unprocessed zipped JSON files:\t", len(zipped_json_files))

    if reproc_csv:
        for zf in zipped_csv_files:
            csv_files = out_csv_file_template.format(
                os.path.split(zf)[1][:-7]).format("*")
            os.system("rm -f {}".format(csv_files))
    if verbose:
        print("Total zipped CSV files:  \t", len(zipped_csv_files))
    zipped_csv_files = [
        zf for zf in zipped_csv_files if not os.path.exists(
            out_csv_file_template.format(os.path.split(zf)[1][:-7]).format(0))
    ]
    if verbose:
        print("Unprocessed zipped CSV files:\t", len(zipped_csv_files))

    def process_zipped_json_file(zf):
        os.system("gunzip -dkf {}".format(zf))
        json_file = zf[:-3]
        process_json_data(
            json_file,
            out_json_file_template.format(os.path.split(json_file)[1][:-5]))
        os.system("rm {}".format(json_file))

    def process_zipped_csv_file(zf):
        os.system("gunzip -dkf {}".format(zf))
        csv_file = zf[:-3]
        process_csv_data(
            csv_file,
            out_csv_file_template.format(os.path.split(csv_file)[1][:-4]))
        os.system("rm {}".format(csv_file))

    # num_jobs = 1
    # joblib.Parallel(n_jobs=num_jobs)(
    #     joblib.delayed(process_zipped_json_file)(i)
    #     for i in tqdm(zipped_json_files[:20]))
    # joblib.Parallel(n_jobs=num_jobs)(joblib.delayed(process_zipped_csv_file)(i)
    #                                  for i in tqdm(zipped_csv_files[:20]))
    num_jobs = 1
    joblib.Parallel(n_jobs=num_jobs)(
        joblib.delayed(process_zipped_json_file)(i)
        for i in tqdm(zipped_json_files[20:400]))
    joblib.Parallel(n_jobs=num_jobs)(joblib.delayed(process_zipped_csv_file)(i)
                                     for i in tqdm(zipped_csv_files[20:400]))
    num_jobs = 8
    joblib.Parallel(n_jobs=num_jobs)(
        joblib.delayed(process_zipped_json_file)(i)
        for i in tqdm(zipped_json_files[400:]))
    joblib.Parallel(n_jobs=num_jobs)(joblib.delayed(process_zipped_csv_file)(i)
                                     for i in tqdm(zipped_csv_files[400:]))


def combine_processed_files(process_dir, verbose=False):
    processed_files = list(glob(os.path.join(process_dir, "*.proc*.tsv")))

    combined_files_dict = {}
    for pf in processed_files:
        cf = os.path.splitext(pf[:-4])[0] + ".tsv"
        if cf in combined_files_dict:
            combined_files_dict[cf].append(pf)
        else:
            combined_files_dict[cf] = [pf]

    for cf, pfs in tqdm(combined_files_dict.items()):
        os.system("touch {}".format(cf))
        for pf in pfs:
            os.system("cat {} >> {}".format(pf, cf))
            os.system("rm {}".format(pf))


def reformat_cdr_range(cdr_range):
    cdr_range = cdr_range.replace(", ", ":").replace("(",
                                                     "[").replace(")", "]")

    return cdr_range


def combine_tsv_files(tsv_files, out_file):
    csv_file = open(out_file, "w")
    col_names = ",".join([
        "seq", "cdr1", "cdr2", "cdr3", "species", "chain_type", "isotype",
        "b_type", "b_source", "disease", "vaccine"
    ])
    csv_file.write(col_names + "\n")

    bad_seqs = 0

    tsv_col_names = [
        "species", "chain_type", "isotype", "b_type", "b_source", "disease",
        "vaccine", "seq", "cdr1_seq", "cdr2_seq", "cdr3_seq", "cdr1_range",
        "cdr2_range", "cdr3_range"
    ]
    use_col_names = [
        "seq", "cdr1_range", "cdr2_range", "cdr3_range", "species",
        "chain_type", "isotype", "b_type", "b_source", "disease", "vaccine"
    ]
    for tsv_file in tqdm(tsv_files):
        # Extract sequence from OAS data

        df = pd.read_csv(tsv_file,
                         delimiter="\t",
                         names=tsv_col_names,
                         usecols=use_col_names)
        df = df.dropna()

        for i, row in df.iterrows():
            seq = row["seq"]
            cdr1 = reformat_cdr_range(row["cdr1_range"])
            cdr2 = reformat_cdr_range(row["cdr2_range"])
            cdr3 = reformat_cdr_range(row["cdr3_range"])
            species = row["species"]
            chain_type = row["chain_type"]
            isotype = row["isotype"]
            b_type = row["b_type"]
            b_source = row["b_source"]
            disease = row["disease"]
            vaccine = row["vaccine"]

            if "[0:-1]" in [cdr1, cdr2, cdr3]:
                bad_seqs += 1
                continue

            csv_components = [
                seq, cdr1, cdr2, cdr3, species, chain_type, isotype, b_type,
                b_source, disease, vaccine
            ]
            csv_line = ",".join(csv_components)
            csv_file.write(csv_line + "\n")

    csv_file.close()
    print(bad_seqs)


def cli():
    project_path = os.path.abspath(os.path.join(deeph3.__file__, "../.."))
    data_path = os.path.join(project_path, "data")

    desc = 'Creates h5 files from all the truncated antibody PDB files in a directory'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        'oas_download_dir',
        type=str,
        help='The directory containing raw zipped CSV and JSON files from OAS')

    parser.add_argument(
        'tsv_out_dir',
        type=str,
        help='The directory containing raw zipped CSV and JSON files from OAS')

    parser.add_argument('--out_file',
                        type=str,
                        default=os.path.join(data_path, 'OAS_unpaired.csv'),
                        help='The name of the outputted CSV file')

    args = parser.parse_args()
    oas_download_dir = args.oas_download_dir
    tsv_out_dir = args.tsv_out_dir
    out_file = args.out_file

    os.system("mkdir {}".format(tsv_out_dir))

    process_zipped_oas_files(oas_download_dir, tsv_out_dir, verbose=True)
    combine_processed_files(tsv_out_dir, verbose=True)

    tsv_files = list(glob(os.path.join(tsv_out_dir, "*.tsv")))
    combine_tsv_files(tsv_files, out_file)


if __name__ == '__main__':
    cli()
