import torch

import matplotlib.pyplot as plt

import numpy as np

import model

import generate_samples

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Device", device)





state_dict = torch.load("./checkpoints/2dFlow")

flow = model.RectifiedFlow(2).to(device)

flow.load_state_dict(state_dict)


c = [1/2,1/2]
means = [[0,10],[20,0]]
variances = [[[1,0],[0,1]],[[3,2],[2,5]]]

c2 = [1/2,1/2]
means2 = [[0,-10],[-20,0]]
variances2 = [[[1,0],[0,1]],[[5,2],[2,3]]]

nsamples = 100

x0 = torch.tensor(generate_samples.get_samples_from_mixed_gaussian(c,means,variances,nsamples)).to(device)
x1 = torch.tensor(generate_samples.get_samples_from_mixed_gaussian(c2,means2,variances2,nsamples)).to(device)


time = np.linspace(0,1,30)

print(time)

sols = []

sols.append(x0.to(device = 'cpu').detach().numpy())

for t in time:
    
    tt = t*torch.ones((nsamples,1),device = x0.device)

    xt = flow(x0,tt)

    xt = xt.to(device = 'cpu').detach().numpy()

    sols.append(xt)

sols= np.array(sols)

sols = np.transpose(sols,(1,0,2))

x0 = x0.to('cpu').detach().numpy()

plt.scatter(x0[:,0],x0[:,1])

plt.scatter(xt[:,0],xt[:,1])
# for trajectory in sols:

#     plt.plot(trajectory[:,0],trajectory[:,1])

plt.show()





