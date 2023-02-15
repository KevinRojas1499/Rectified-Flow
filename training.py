import torch 
import model

def train(x0,x1, nump_epochs,path_to_save,flow_network : model.RectifiedFlow):
    optimizer = torch.optim.SGD(flow_network.parameters(),lr=.01)
    losses = torch.zeros(nump_epochs)
    for i in range(nump_epochs):
        optimizer.zero_grad()
        loss = loss_function(x0,x1,flow_network)
        loss.backward()
        losses[i] = loss
        optimizer.step()
        
        if i%10000 == 0:
            print(f"Loss {loss}")
            torch.save(flow_network.state_dict(), path_to_save)


    return [] 



def loss_function(x0,x1, flow):
    random_t = torch.rand((1000,1))

    xt = random_t*x1 +(1-random_t)*x0 

    vxt = flow(xt,random_t) 

    return torch.mean((x1-x0-vxt)**2)
