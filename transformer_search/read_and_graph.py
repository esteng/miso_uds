import numpy as np
from matplotlib import pyplot as plt 
import glob 

def read_file(path):
    with open(path) as f1:
        return f1.readlines()

def find_best(lines):
    find_text = "Best validation performance so far." 
    find_f1 = "allennlp.training.tensorboard_writer - s_f1" 
    for i in range(len(lines)-1, -1, -1):
        found = False  
        if find_text in lines[i]:
            for j in range(i-1, i - 19, -1):
                if find_f1 in lines[j]:
                    f1_score = parse_line(lines[j]) 
                    found = True
                    break
        if found:
            break
            

    return f1_score

def parse_line(line): 
    return float(line.strip()[-7:])

def parse_name(log_name):
    return log_name.split("/")[-2]

def parse_all(): 
    to_plot = {}
    for log in glob.glob("/exp/estengel/miso_res/tuning/*/stdout.log"): 
        try:
            lines = read_file(log)
            best = find_best(lines)
            name = parse_name(log) 
            to_plot[name] = best
        except:
            continue
    return to_plot

def graph(to_plot): 
    to_plot_items = sorted(list(to_plot.items()), key = lambda x: x[1]) 
    print(to_plot_items[-10:]) 
    to_plot_labels, to_plot_ys = zip(*to_plot_items) 
    to_plot_xs = [i for i in range(len(to_plot_ys))] 
    print(f"best config {to_plot_labels[-1]} with value {to_plot_ys[-1]}") 

    plt.plot(to_plot_xs, to_plot_ys, 'o') 
    plt.xticks(ticks = to_plot_xs, rotation = 45, labels = to_plot_labels) 
    plt.tight_layout() 
    plt.savefig("results.png") 
        
if __name__ == "__main__": 
    #lines = read_file("/exp/estengel/miso_res/tuning/128-10-16-0.5-4000.ckpt/stdout.log")
    #best = find_best(lines)
    #assert(best == 74.551) 
    #print("done")     
    to_plot = parse_all() 
    graph(to_plot) 
