import re 

def is_english_punct(c):
    return re.search(r'^[,.?!:;"\'-(){}\[\]]$', c)
