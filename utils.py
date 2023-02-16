import matplotlib.pyplot as plt

import model
import torch
def plot_connection_lines(x0,x1):
    for i in range(x0.shape[0]):
        plt.plot((x0[i,0],x1[i,0]),(x0[i,1],x1[i,1]),color='green')

def plot_flow(x0,x1):
    plt.scatter(x0[:,0],x0[:,1],color='blue')
    plt.scatter(x1[:,0],x1[:,1],color='orange')
    plot_connection_lines(x0,x1)
    plt.show()

def simulate_ode(x0, flow_network : model.RectifiedFlow):
    time_steps = 1000
    h = 1/time_steps
    xt = x0.clone()
    t = torch.ones((x0.shape[0],1),device=x0.device)*h
    for i in range(time_steps):
        xt += flow_network(xt,t)*h
        t+=h
    return xt