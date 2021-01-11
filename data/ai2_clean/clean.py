import pathlib


paths = pathlib.Path("").glob("*.conllu") 

for path in paths:
    with open(path) as f1:
        lines = f1.readlines()
    # filter 
    lines = [ line for line in lines if not line.startswith("#")]
    with open(path, "w") as f1:
        for line in lines:
            f1.write(line) 
