import numpy as np
import matplotlib.pyplot as plt

def generate_sinus(noise=True):
    signal = []
    for i in range(5):
        x = np.linspace(-np.pi, np.random.randint(1,6)*np.pi,np.random.randint(100,1000))
        # Generate different frequency
        signal.extend(list(np.sin(np.random.randint(1,2)*x)))
        # add noise
    if noise == True:
        signal =np.random.randint(5,20)*np.array(signal) + 3*np.random.rand(len(signal))
    else:
        signal=np.random.randint(5,20)*np.array(signal)
    return signal

def generate_dataset(noise=True,trend=True, error=True):
    data_list = [[]]
    for i in range(5):
        x = generate_sinus(noise)
        for j in range(40):
            if error == True:
                x[np.random.randint(0,len(x))] =np.random.randint(-20,20)
            if trend == True:
                # # Line
                # x = x + np.linspace(0,2,len(x))
                # # broken line
                # x = x +np.concatenate((np.linspace(0,2,len(x)//2),np.linspace(2,0,len(x)-len(x)//2)))
                # # hyperbola
                # x = x + np.linspace(0,1.4,len(x))**2
                # sudden mean shift
                x = x + np.concatenate((np.linspace(0, 1, len(x) // 2), np.linspace(0.7, 1, len(x) - (len(x) // 2))))
        data_list[0].append(x)
    data = np.array(data_list)
    return data

def show():
    data = generate_dataset()
    for i in range(len(data[0])):
        plt.figure()
        plt.plot(data[0][i])
show()