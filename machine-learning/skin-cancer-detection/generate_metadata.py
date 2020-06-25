import glob
import os
import argparse

import pandas as pd


def generate_csv(folder, labels):
    folder_name = os.path.basename(folder)
    # convert comma separated labels into a list
    label2int = {}
    if labels:
        labels = labels.split(",")
        for label in labels:
            string_label, integer_label = label.split("=")
            label2int[string_label] = integer_label

    labels = list(label2int)
    # generate CSV file
    df = pd.DataFrame(columns=["filepath", "label"])
    i = 0
    for label in labels:
        print("Reading", os.path.join(folder, label, "*"))
        for filepath in glob.glob(os.path.join(folder, label, "*")):
            df.loc[i] = [filepath, label2int[label]]
            i += 1

    df.to_csv(f"{folder_name}.csv")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CSV Metadata generator for skin cancer dataset from ISIC")
    parser.add_argument("-f", "--folder", help="Dataset portion folder, e.g: /root/skin-disease/test and not the whole dataset", 
                        required=True)
    parser.add_argument("-l", "--labels", help="The different skin disease classes along with label encoding separated in commas, \
                        e.g: Binary classification between malignant and benign categories, something like this: \
                        nevus=0,seborrheic_keratosis=0,melanoma=1",
                        required=True)
    # parse arguments
    args = parser.parse_args()
    folder = args.folder
    labels = args.labels
    # generate the CSV file
    generate_csv(folder, labels)

