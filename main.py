import torch 
import generate_samples
import training
import model 
import matplotlib.pyplot as plt

c = [1/2,1/2]
means = [[0,10],[20,0]]
variances = [[[1,0],[0,1]],[[3,2],[2,5]]]

c2 = [1/2,1/2]
means2 = [[0,-10],[-20,0]]
variances2 = [[[1,0],[0,1]],[[5,2],[2,3]]]

x0 = torch.tensor(generate_samples.get_samples_from_mixed_gaussian(c,means,variances,1000))
x1 = torch.tensor(generate_samples.get_samples_from_mixed_gaussian(c2,means2,variances2,1000))

# plt.scatter(x0[:,0],x0[:,1])
# plt.scatter(x1[:,0],x1[:,1])
# plt.show()

path_to_save = './checkpoints/2dFlow'
num_epochs = 150000
flow_network = model.RectifiedFlow(2)

training.train(x0=x0,
               x1=x1,
               nump_epochs=num_epochs,
               flow_network=flow_network,
               path_to_save=path_to_save)




