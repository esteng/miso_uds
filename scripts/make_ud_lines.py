import pathlib 
import sys 

def get_blocks(ud_str):
    blocks = ud_str.split("\n\n")[:-1]
    blocks = [block.split("\n") for block in blocks]
    return blocks

def get_forms(blocks):
    sentences = []
    all_tags = []
    for block in blocks:
        sent = []
        tags = []
        for line in block:
            if line[0] == "#":
                continue
            form = line.split("\t")[1]
            tag = line.split("\t")[3]
            sent.append(form)
            tags.append(tag)
        sentences.append((" ".join(sent), ",".join(tags)))
    return sentences
    
# get all possible "with" pp ambiguity sentences 
def read_ud_input(path):
    with open(path) as f1:
        data_str = f1.read()
        data_blocks = get_blocks(data_str)
        data_sents = get_forms(data_blocks)

    return data_sents

if __name__ == "__main__": 
    top = pathlib.Path(sys.argv[1]) 
    clean = pathlib.Path(sys.argv[2]) 
    all_ud = top.glob("*.conllu")
    for path in all_ud:     
        print(str(path))
        data = read_ud_input(str(path)) 
        filename = path.stem + ".lines"
        with open(clean.joinpath(filename), "w") as f1:
            for sent, tags in data:
                f1.write(f"{sent}\t{tags}\n")
