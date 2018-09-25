#!/usr/bin/env python
import argparse
import json
import sys

def write_json(
        file_to_write,
        stacked_lines,
        sentence_id
):
    """
    This function is to make json string to be dumped
    :param file_to_write: could be None. If None, print to stdout
    :param stacked_lines: stacked split lines from conllu file
    :param sentence_id: sentence ID
    :return:
    """
    dict_to_write = {}
    dict_to_write['sent_id'] = sentence_id
    dict_to_write['tokens'] = [item[1]  for item in stacked_lines]
    dict_to_write['lemma'] = [item[2]  for item in stacked_lines]
    dict_to_write['pos'] = [item[3]  for item in stacked_lines]
    dict_to_write['relations'] = [(item[0], item[6], item[7]) for item in stacked_lines]

    line_to_write = json.dumps(dict_to_write) + '\n'

    if file_to_write:
        with open(file_to_write, 'a') as f:
            f.write(line_to_write)
    else:
        sys.stdout.write(line_to_write)



if __name__ == "__main__":
    parser = argparse.ArgumentParser("conllu2json")

    parser.add_argument("--conllu", "-c",
                        help='Input conllu file')
    parser.add_argument("--json", "-j",
                        help='Output json file. If not provided, the output will be push to stdout')

    opts = parser.parse_args()

    # Whether use argument or stdin
    if opts.conllu:
        f_in = open(opts.conllu)
    else:
        f_in = sys.stdin
        sys.stderr.write("No input file or can't touch a output file, will read from stdin\n")

    # Try to touch the output file
    if opts.json:
        with open(opts.json, 'w') as f_out:
            pass
    else:
        sys.stderr.write("No output file or can't touch a output file, the results will be in stdout\n")


    # skip doc id line
    f_in.readline()

    stacked_lines = []
    sentence_id = None
    sentence_conter = 0

    for line in f_in:
        if len(line) <= 1:
            write_json(
                opts.json,
                stacked_lines,
                sentence_id
            )

            stacked_lines = []
            sentence_id = None
            sentence_conter += 1

        else:
            tuples = line.strip().split()
            if tuples[0] == "#":
                if tuples[1] == "sent_id":
                    sentence_id = tuples[-1]
                    stacked_lines = []
            else:
                stacked_lines.append(tuples)

    if opts.conllu:
        f_in.close()

    sys.stderr.write("Convert {} sentences".format(sentence_conter) + '\n')

