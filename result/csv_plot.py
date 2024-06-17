import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def csv_plot(file_name, real_x, real_z): 
    df = pd.read_csv(file_name)
    df = pd.DataFrame(df, columns=['x', 'z'])
    data = df.to_numpy()
    x = data[:, 0]
    z = data[:, 1]
    plt.plot(x, z, label="simulation")
    plt.plot(real_x, real_z, '-', label="real")
    plt.gca().set_aspect(1)
    plt.xlim([np.min(x)-0.01, np.max(x)+0.01])
    plt.ylim([np.min(z)-0.01, np.max(z)+0.01])
    plt.xlabel("x (unit: m)")
    plt.ylabel("z (unit: m)")
    plt.title("x-z plane")
    plt.legend(loc="upper right")
    plt.show()
    

if __name__ == "__main__":
    circle = False
    if circle:
        file_name = "circle.csv"
        t = np.array([(theta * np.pi / 180) for theta in range(360)])
        x = np.cos(t) * 0.05 + 0
        z = np.sin(t) * 0.05 + 0.1
    else:
        file_name = "line.csv"
        x = np.array([i / 100 for i in range(15)])
        z = -x / 3 + 0.15

    csv_plot(file_name, x, z)