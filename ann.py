import torch
import torch.nn as nn 
import torchvision
import torchvision.transforms as trans
import matplotlib.pyplot as plot 
inpsiz = 784 
hidensiz = 120 
numclases = 10
numepchs = 4
bachsiz = 100
l_r = 0.001 

trainds = torchvision.datasets.MNIST(root='./data', 
                                          train=True, 
                                          transform=trans.ToTensor(),  
                                          download=True)
testds = torchvision.datasets.MNIST(root='./data', 
                                           train=False, 
                                           transform=trans.ToTensor()) 
length = len(trainds)
trainldr = torch.utils.data.DataLoader(dataset=trainds, 
                                           batch_size=bachsiz, 
                                           shuffle=True)
testldr = torch.utils.data.DataLoader(dataset=testds, 
                                           batch_size=bachsiz, 
                                           shuffle=False)

class neural_network(nn.Module):
    def __init__(self, inpsiz, hidensiz, numclases):
         super(neural_network, self).__init__()
         self.inputsiz = inpsiz
         self.l1 = nn.Linear(inpsiz, hidensiz) 
         self.relu = nn.ReLU()
         self.l2 = nn.Linear(hidensiz, numclases) 
    def forward(self, y):
         outp = self.l1(y)
         outp = self.relu(outp)
         outp = self.l2(outp)

         return outp
modl = neural_network(inpsiz, hidensiz, numclases)

criter = nn.CrossEntropyLoss()
optim = torch.optim.Adam(modl.parameters(), lr=l_r)
nttlstps = len(trainldr)
for epoch in range(numepchs):
    score = 0
    loss = 0
    for x, (imgs, lbls) in enumerate(trainldr): 
        imgs = imgs.reshape(-1, 28*28)
        labls = lbls

        outp = modl(imgs)
        losses = criter(outp, lbls)
        loss += losses.item()
        optim.zero_grad()
        losses.backward()
        optim.step() 

        inference = torch.argmax(outp, axis=1)
        score += torch.sum(torch.eq(inference, lbls))
    print (f'Epochs [{epoch+1}/{numepchs}]], Acc: {score.item()/length *100 :.4f}, loss: {loss}')