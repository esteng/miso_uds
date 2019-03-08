import sys

align = []
with open(sys.argv[1]) as f:
    for line in f:
        align.append(
            {int(item.split("-")[1]) : int(item.split("-")[0]) for item in line.strip().split()}
        )

with open(sys.argv[2]) as f:
    for line_idx, line in enumerate(f):
        src_string, tgt_string = line.strip().split(" ||| ")

        src_tokens = src_string.split()
        tgt_tokens = tgt_string.split()
        
        check_list = []

        for idx, token in enumerate(tgt_tokens):
            if idx in align[line_idx]:
                check_list.append(token + ">>" + src_tokens[align[line_idx][idx]])
            else:
                check_list.append(token)

        print(" ".join(check_list))





