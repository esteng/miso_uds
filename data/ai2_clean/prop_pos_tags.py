import pathlib
import sys
import pdb 

def prop(in_block, out_block):
    in_block = [x for x in in_block if not x.startswith("#") ]
    out_block = [x for x in out_block if not x.startswith("#") ]

    for i, (in_line, out_line) in enumerate(zip(in_block, out_block)): 
        in_line, out_line = in_line.split("\t"), out_line.split("\t") 
        try:
            out_line[3] = in_line[3]
            out_line[4] = in_line[4]
        except IndexError:
            pdb.set_trace() 

        out_line = "\t".join(out_line) 
        out_block[i] = out_line
    return out_block

def read_blocks(ud_path):
    with open(ud_path) as f1:
        ud_str = f1.read() 
    blocks = ud_str.split("\n\n")[:-1]
    blocks = [block.split("\n") for block in blocks]
    blocks = [[x for x in block if x != ''] for block in blocks]
    return blocks

if __name__ == "__main__":
    in_path = sys.argv[1]
    pred_path = sys.argv[2]
    
    pred_dir_path = pathlib.Path(pred_path)
    pred_dir_path, filename = pred_dir_path.parent, pred_dir_path.stem 
    out_path = pred_dir_path.joinpath(filename + ".propped.conllu")

    in_blocks = read_blocks(in_path)
    pred_blocks = read_blocks(pred_path) 
    out_blocks = [] 
    for i, (in_block, pred_block) in enumerate(zip(in_blocks, pred_blocks)):
        out_block = prop(in_block, pred_block) 
        out_blocks.append(out_block) 

    out_blocks = ["\n".join(block) for block in out_blocks]
    out_blocks = "\n\n".join(out_blocks)
    with open(out_path, "w") as f1:
        f1.write(out_blocks) 
        
    
    
