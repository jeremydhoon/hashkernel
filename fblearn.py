#!/usr/bin/env python

"""
fblearn -- train a hashkernel classifier on heterogenous Facebook data objects.
"""

import argparse
import copy
import json
import os
import shutil
import sys
import urllib2

import hashkernel

def get_url(field, access_token):
    return ("https://graph.facebook.com/me/%s?access_token=%s "
            % (field, access_token))

def fb_data_iterator(access_token, output_filename, fields=None):
    fields = fields or ["photos"]
    if os.path.exists(output_filename):
        shutil.copy(output_filename, output_filename + ".bak")
    with open(output_filename, "wb") as outfile:
        outfile.write("[\n")
        saw_prev = False
        for field in fields:
            url = get_url(field, access_token)
            while True:
                try:
                    doc = urllib2.urlopen(url, timeout=100000000)
                    feed = json.load(doc)
                    if "paging" not in feed:
                        break
                    url = feed["paging"]["next"]
                    if saw_prev:
                        outfile.write(",\n")
                    outfile.write(json.dumps(feed["data"], indent=4))
                    saw_prev = True
                except KeyboardInterrupt:
                    break
        outfile.write("]\n")

def load_fb_data(filename):
    return sum(json.load(open(filename, "r")), [])

def is_liked(item, threshold):
    if "likes" not in item:
        return False
    elif "count" not in item["likes"]:
        item["likes"]["count"] = len(item["likes"]["data"])
    likes = item.pop("likes")
    return likes["count"] >= threshold

def extract_labels(fb_data, threshold):
    return [(item, is_liked(item, threshold))
            for item in copy.deepcopy(fb_data)]

def train_test(instances, test_index, bits, salts):
    kernel = hashkernel.HashKernelLogisticRegression(
        bits=bits,
        salts=range(salts))
    test_instance, test_label = instances[test_index]
    for ix, (item, label) in enumerate(instances):
        if ix == test_index:
            continue
        kernel.add(item, label)
    return kernel.predict(test_instance), test_label

def main(argv):
    assert argv, "Invalid command line arguments"

    access_token_help = (
        "If specified, use this access token to load Facebook data. "
        "To obtain an access token, register as a Facebook Developer and "
        "visit https://developers.facebook.com/tools/explorer.")
    fb_file_help = (
        "File in which to read/write Facebook data. If an access token is "
        "also provided, then overwrites this file and subsequently reads "
        "its contents.")

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--access-token", dest="access_token",
                        action="store", type=str, default=None,
                        help=access_token_help)
    parser.add_argument("-f", "--fb-file", dest="fb_file",
                        action="store", type=str, default=None,
                        help=fb_file_help)
    parser.add_argument("-d", "--fields", dest="fields", action="store",
                        type=str, default="statuses",
                        help=("Types of FB data to fetch (only valid with "
                              "an access token). Comma-separated."))
    parser.add_argument("-n", "--like-threshold", dest="like_threshold",
                        action="store", type=int, default=1,
                        help=("Minimum number of likes an object must "
                              "recieve in order to be considered a positive "
                              "instance."))
    parser.add_argument("-s", "--salts", dest="salts", type=int, default=1,
                        help="Number of salts to use.")
    parser.add_argument("-b", "--bits", dest="bits", action="store", type=int,
                        default=14, help="Number of hash kernel bits to use.")
    args = parser.parse_args(argv[1:])

    if args.fb_file is None:
        args.fb_file = os.path.join(
            os.path.dirname(__file__),
            "data",
            "feed.json")
    if args.access_token:
        fields = args.fields.split(",")
        fb_data_iterator(args.access_token, args.fb_file, fields)
    if not os.path.exists(args.fb_file):
        parser.error(
            ("FB data file does not exist: %s. "
             "Supply an access token to generate a data file.") % args.fb_file)
        return 1

    fb_data = load_fb_data(args.fb_file)
    labeled_fb_data = extract_labels(fb_data, args.like_threshold)
    false_positive = 0
    false_negative = 0
    count = len(fb_data)
    num_liked = sum([int(label) for _, label in labeled_fb_data])
    for test_index in xrange(count):
        pred, label = train_test(
            labeled_fb_data,
            test_index,
            args.bits,
            args.salts)
        if pred and not label:
            false_positive += 1
        elif label and not pred:
            false_negative += 1
    print "failure rate:", float(false_positive + false_negative) / count
    print "fp/fn/total", false_positive, false_negative, count
    print "rate liked:", float(num_liked) / count


if __name__ == "__main__":
    sys.exit(main(sys.argv))
