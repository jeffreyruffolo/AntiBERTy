import numpy as np
import argparse
from tqdm import tqdm
import os
from glob import glob
import pandas as pd
import json
import itertools

import deeph3

h_components = ["fwh1", "cdrh1", "fwh2", "cdrh2", "fwh3", "cdrh3", "fwh4"]
h_cdr_names = ["h1", "h2", "h3"]
l_components = ["fwl1", "cdrl1", "fwl2", "cdrl2", "fwl3", "cdrl3", "fwl4"]
l_cdr_names = ["l1", "l2", "l3"]


def to_dict(text):
    return json.loads(text.replace("\'", "\""))


def combine_anarci_components(anarci_dict, components):
    seq_list = list(
        itertools.chain.from_iterable(
            [list(anarci_dict[c].values()) for c in components]))
    seq = "".join(seq_list)

    return seq


def extract_seq_components(anarci_dict, seq_components, cdr_names):
    seq = combine_anarci_components(anarci_dict, seq_components)

    cdr_range_dict = {}
    cdr_seq_dict = {}
    for i in range(0, 3):
        cdr_range_dict[cdr_names[i]] = [
            len(
                combine_anarci_components(anarci_dict,
                                          seq_components[:2 * i + 1])),
            len(
                combine_anarci_components(anarci_dict,
                                          seq_components[:2 * i + 2])) - 1
        ]
        cdr_seq_dict[cdr_names[i]] = combine_anarci_components(
            anarci_dict, ["cdr" + cdr_names[i]])

    return seq, cdr_range_dict, cdr_seq_dict


def process_csv_data(csv_file, verbose=False):
    rep_info = to_dict(
        np.genfromtxt(csv_file,
                      max_rows=1,
                      dtype=str,
                      delimiter="\t",
                      comments=None).item())
    info_dict = {
        "species": rep_info["Species"],
        "isotype": rep_info["Isotype"],
        "b_type": rep_info["BType"],
        "b_source": rep_info["BSource"],
        "disease": rep_info["Disease"],
        "vaccine": rep_info["Vaccine"],
    }

    col_names = pd.read_csv(csv_file, skiprows=1, nrows=1).columns
    max_rows = int(1e6)
    oas_df = pd.read_csv(csv_file,
                         skiprows=1,
                         names=col_names,
                         header=None,
                         usecols=[
                             'ANARCI_status_light', 'ANARCI_status_heavy',
                             'ANARCI_numbering_heavy', 'ANARCI_numbering_light'
                         ])
    oas_df = oas_df.query(
        "ANARCI_status_light == 'good' and ANARCI_status_heavy == 'good'")
    oas_df = oas_df[['ANARCI_numbering_heavy', 'ANARCI_numbering_light']]

    data_list = []
    for index, (anarci_h_data, anarci_l_data) in enumerate(oas_df.values):
        missing_component = False
        for c in h_components:
            if not c in to_dict(anarci_h_data):
                if verbose:
                    print("Missing heavy component in index {}: {}".format(
                        index, c))
                missing_component = True
        for c in l_components:
            if not c in to_dict(anarci_l_data):
                if verbose:
                    print("Missing light component in index {}: {}".format(
                        index, c))
                missing_component = True

        if missing_component:
            continue

        heavy_prim, h_cdr_range_dict, h_cdr_seq_dict = extract_seq_components(
            to_dict(anarci_h_data), h_components, h_cdr_names)
        light_prim, l_cdr_range_dict, l_cdr_seq_dict = extract_seq_components(
            to_dict(anarci_l_data), l_components, l_cdr_names)

        data_list.append({
            "heavy_data": (heavy_prim, h_cdr_range_dict, h_cdr_seq_dict),
            "light_data": (light_prim, l_cdr_range_dict, l_cdr_seq_dict),
            "metadata":
            info_dict
        })

    return data_list


def sequences_to_csv(oas_csv_dir,
                     out_file,
                     print_progress=False,
                     verbose=False):

    oas_csv_files = glob(os.path.join(oas_csv_dir, "*.csv"))
    data_list = []
    for oas_csv in tqdm(oas_csv_files):
        data_list.extend(process_csv_data(oas_csv, verbose=verbose))

    num_seqs = len(data_list)
    max_h_len = 200
    max_l_len = 200

    csv_file = open(out_file, "w")
    col_names = ",".join([
        "hseq", "lseq", "h1", "h2", "h3", "l1", "l2", "l3", "species",
        "isotype", "b_type", "b_source", "disease", "vaccine"
    ])
    csv_file.write(col_names + "\n")

    for index, data_dict in tqdm(enumerate(data_list),
                                 disable=(not print_progress)):
        # Extract sequence from OAS data
        heavy_prim, h_cdr_range_dict, h_cdr_seq_dict = data_dict["heavy_data"]
        light_prim, l_cdr_range_dict, l_cdr_seq_dict = data_dict["light_data"]
        metadata = data_dict["metadata"]

        # heavy_prim = " ".join(list(heavy_prim))
        # light_prim = " ".join(list(light_prim))

        cdr_range_dict = {}
        cdr_range_dict.update(h_cdr_range_dict)
        cdr_range_dict.update(l_cdr_range_dict)

        cdr_ranges = [
            cdr_range_dict["h1"], cdr_range_dict["h2"], cdr_range_dict["h3"],
            cdr_range_dict["l1"], cdr_range_dict["l2"], cdr_range_dict["l3"]
        ]
        cdr_ranges = [str(cdr_r).replace(",", ":") for cdr_r in cdr_ranges]

        csv_components = [
            heavy_prim, light_prim, *cdr_ranges, metadata["species"],
            metadata["isotype"], metadata["b_type"], metadata["b_source"],
            metadata["disease"], metadata["vaccine"]
        ]
        csv_line = ",".join(csv_components)
        csv_file.write(csv_line + "\n")

    csv_file.close()


def cli():
    project_path = os.path.abspath(os.path.join(deeph3.__file__, "../.."))
    data_path = os.path.join(project_path, "data")

    desc = 'Creates h5 files from all the truncated antibody PDB files in a directory'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        'oas_csv_dir',
        type=str,
        help=
        'The directory containing antibody sequence CSV files downloaded from OAS'
    )
    parser.add_argument('--out_file',
                        type=str,
                        default=os.path.join(data_path, 'OAS_paired.csv'),
                        help='The name of the outputted CSV file')

    args = parser.parse_args()
    oas_csv_dir = args.oas_csv_dir
    out_file = args.out_file

    sequences_to_csv(oas_csv_dir, out_file, print_progress=True)


if __name__ == '__main__':
    cli()
