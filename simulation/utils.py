import pickle as pkl

def read_pkl(file_name):
    with open(file_name, 'rb') as f:
        data = pkl.load(f)  
    return data