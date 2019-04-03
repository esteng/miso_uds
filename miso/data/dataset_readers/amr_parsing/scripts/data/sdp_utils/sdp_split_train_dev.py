import argparse
import os

def lazy_load(text):
    for sentence in text.split("\n\n"):
        yield sentence

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-section", default=20, type=int,
                        help="Section for dev")
    parser.add_argument("--input", required=True,
                        help="Original data file")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory")
    parser.add_argument("--skip-first-line", action="store_true",
                        help="Output directory")
    args = parser.parse_args()

    suffix = os.path.basename(args.input)

    input_file = open(args.input, 'r')

    train_file = open(os.path.join(args.output_dir, "train." + suffix), 'w+') 
    dev_file = open(os.path.join(args.output_dir, "dev." + suffix), 'w+') 

    if args.skip_first_line:
        firstline = input_file.readline()
        dev_file.write(firstline)
        train_file.write(firstline)

    for sentence in lazy_load(input_file.read()):
        if len(sentence) == 0:
            continue

        section_id = int(sentence[2:4])
        if section_id == args.dev_section:
            dev_file.write(sentence + '\n\n')
        else:
            train_file.write(sentence + '\n\n')

    


