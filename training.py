import torch 
import model

def train(x0,x1, nump_epochs,path_to_save,flow_network : model.RectifiedFlow):
    # optimizer = torch.optim.SGD(flow_network.parameters(),lr=3e-4)
    optimizer = torch.optim.Adam(flow_network.parameters(), lr=3e-4)

    losses = torch.zeros(nump_epochs)
    for i in range(nump_epochs):
        optimizer.zero_grad()
        loss = loss_function(x0,x1,flow_network)
        loss.backward()
        losses[i] = loss
        optimizer.step()
        
        if i%10000 == 0:
            print(f"Loss {i//10000} {loss}")
            torch.save(flow_network.state_dict(), path_to_save)


    return [] 



def loss_function(x0,x1, flow : model.RectifiedFlow):
    random_t = torch.rand((x0.shape[0],1),device=x0.device)

    xt = x0+random_t*(x1-x0) 

    vxt = flow(xt,random_t) 

    return torch.mean((x1-x0-vxt)**2)
