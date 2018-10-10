import matplotlib.pyplot as plt

def data_compare(inputs, outputs, title='', interval=(0,6000), legends=['input','output']):
    """Create subplots"""
    plt.figure()
    plt.title(title)
    plt.subplot(311)
    plt.plot(inputs[slice(*interval)])
    plt.plot(outputs[slice(*interval)])
    plt.legend(legends)

    plt.subplot(312)
    plt.plot(inputs[slice(*interval)])
    plt.legend(legends[0])

    plt.subplot(313)
    plt.plot(outputs[slice(*interval)])
    plt.legend(legends[1])

    # Remark: ion() turns the interactive mode on, unless the program will be blocked when the figure shows
    # But is only used for debug/interactive mode
    plt.ion()
    plt.show()
    plt.pause(0.001)

def single_plot(data_list, path, title='', legends=''):
    """Plot the signals in the same figure"""
    plt.figure()
    for data in data_list:
        plt.plot(data)
    plt.legend(legends)
    plt.title(title)
    plt.ion()
    plt.show()
    plt.pause(0.001)
    plt.savefig(path)