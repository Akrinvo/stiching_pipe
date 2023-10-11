import numpy as np
import matplotlib.pyplot as plt
# import json
def predict_points(d1):
    d = np.array([20, 25, 30, 35, 40, 45, 50,52.66])
    
    x_1 = np.array([0.32542,0.28178,0.23814,0.19449,0.15085,0.10720,0.06356,0.04034])

    y_2 = np.array([0.13159,0.12209,0.11209,0.10154,0.09040,0.07863,0.06615,0.05921])
    

    func_up = np.polyfit(d, x_1, 1)
    func1 = np.poly1d(func_up)
    x1 = func1(d1)

    func_low = np.polyfit(d, y_2, 1)
    func2 = np.poly1d(func_low)
    y2 = func2(d1)
    plt.plot(d,x_1)
    plt.show()
    plt.plot(d,y_2)
    plt.show()
    
    return x1,y2
if __name__=="__main__":
    d1 = float(input("Enter D: "))
    x1,y2 = predict_points(d1)
    print(x1,y2)

