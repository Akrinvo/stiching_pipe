import numpy as np
import matplotlib.pyplot as plt
# import json
def predict_points(d1):
    d = np.array([20, 25, 30, 35, 40, 40.36, 45, 50,52.66, 55,28.42])
    
    x_1 = np.array([0.31745, 0.27181, 0.26618, 0.18053,
                        0.13490, 0.13161, 0.08926, 0.04362, 0.01934, -0.00202,0.24114])

    y_2 = np.array([0.12069, 0.11109, 0.10106, 0.09055,
                        0.07956, 0.07874, 0.06802, 0.05590, 0.04921, 0.04317,0.10428])
    

    func_up = np.polyfit(d, x_1, 1)
    func1 = np.poly1d(func_up)
    x1 = func1(d1)

    func_low = np.polyfit(d, y_2, 1)
    func2 = np.poly1d(func_low)
    y2 = func2(d1)
    plt.plot(d,x_1,"+")
    plt.show()
    
    return x1,y2
if __name__=="__main__":
    d1 = float(input("Enter D: "))
    x1,y2 = predict_points(d1)
    print(x1,y2)

