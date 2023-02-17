import matplotlib.pyplot as plt
import torch

import generate_samples
import model
import training
import utils 



c = [1/2,1/2]
means = [[0,10],[20,0]]
variances = [[[1,0],[0,1]],[[3,2],[2,5]]]

c2 = [1/2,1/2]
means2 = [[0,-10],[-20,0]]
variances2 = [[[1,0],[0,1]],[[5,2],[2,3]]]

num_samples = 100

############ Data generation
torch.random.seed()
x0 = torch.tensor(generate_samples.get_samples_from_mixed_gaussian(c,means,variances,num_samples))
x1 = torch.tensor(generate_samples.get_samples_from_mixed_gaussian(c2,means2,variances2,num_samples))

utils.plot_flow(x0,x1)

############ Set up Variables
path_to_save = './checkpoints/2dFlow'
num_epochs = 150001

############ Move everything to GPU 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
flow_network = model.RectifiedFlow(2).to(device=device)
x0 = x0.to(device=device)
x1 = x1.to(device=device)

############ Training Loop
train = False
if train:
    training.train(x0=x0,
                x1=x1,
                nump_epochs=num_epochs,
                flow_network=flow_network,
                path_to_save=path_to_save)

############ Load Network if checkpoint is available

checkpoint = torch.load(path_to_save)
flow_network.load_state_dict(checkpoint)

z1 = utils.simulate_ode(x0,flow_network)

x0 = x0.to('cpu').detach().numpy()
x1 = x1.to('cpu').detach().numpy()
z1 = z1.to('cpu').detach().numpy()

utils.plot_flow(x0,z1)

