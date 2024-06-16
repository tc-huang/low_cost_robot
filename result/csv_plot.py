import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def csv_plot(file_name): 
    df = pd.read_csv(file_name)
    df = pd.DataFrame(df, columns=['x', 'z'])
    data = df.to_numpy()
    x = data[:, 0]
    z = data[:, 1]
    plt.plot(x, z)
    plt.gca().set_aspect(1)
    plt.xlim([np.min(x)-0.01, np.max(x)+0.01])
    plt.ylim([np.min(z)-0.01, np.max(z)+0.01])
    plt.xlabel("x (unit: m)")
    plt.ylabel("z (unit: m)")
    plt.title("x-z plane")
    plt.show()
    

if __name__ == "__main__":
    file_name = "circle.csv"
    file_name = "line.csv"
    
    csv_plot(file_name)