import matplotlib.pyplot as plt

def loss_graph(loss_list):
    plt.title("Average difference over times")
    plt.xlabel("Times")
    plt.ylabel("Average difference")
    x = []
    for i in range(len(loss_list)):
        x.append(i)
    plt.plot(x, loss_list, label="Loss function")
    plt.show(block=False)
    plt.waitforbuttonpress()
    plt.clf()
    plt.close()
    return -1