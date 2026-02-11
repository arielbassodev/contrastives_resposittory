from augmentation import *
from tqdm import tqdm
import matplotlib.pyplot as plt
from contrastive_approaches import *
from torchvision.models import resnet50
import torch.optim as optim
import lightly
from lightly.loss import NTXentLoss
import torchvision.datasets as datasets
from utils import *
device = ("cuda" if torch.cuda.is_available() else "cpu")



def Simclr_trainer(model, train_loader, criterion, optimizer, epochs, active_groups):
  epoch_losses = []
  
  for epoch in range(epochs): 
    running_loss = 0 
    for step, (data, _) in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()
        first_batch, second_batch = global_global_augmentation(data, active_groups, device)
        embd_batch_1 = model(first_batch)
        embd_batch_2 = model(second_batch)
        loss = criterion(embd_batch_1, embd_batch_2)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()        
    epoch_loss = running_loss / len(train_loader)
    print(epoch_loss)
    epoch_losses.append(epoch_loss)
  plot_training_loss(epoch_loss, epochs)
        
def Slfpn_trainer(model, projector, train_loader, criterion, optimizer, epochs, active_groups):
    epoch_losses = []
  
    for epoch in range(epochs):
        running_loss = 0.0
        print(f"Début de l'époque {epoch + 1}")
        for step, (data, label) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            batch_1, batch_2 = global_global_augmentation(data, active_groups)
            batch_1, batch_2 = batch_1.to('cuda'), batch_2.to('cuda')
            data = data.to('cuda')
            embedding_batch_1 = model(batch_1)
            embedding_batch_2 = model(batch_2)
            embedding = model(data)
            loss_1 = criterion(embedding_batch_1,embedding_batch_2)
            loss_2 = criterion(embedding, embedding_batch_2)
            original_projection = projector(embedding)
            loss_3 = criterion(original_projection, embedding_batch_2.detach())
            loss = loss_1 + loss_2 + loss_3
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_loader)
        print(epoch_loss)
        epoch_losses.append(epoch_loss)
    plot_training_loss(epoch_loss, epochs)
    
import matplotlib.pyplot as plt



# # train Slf with an MSE loss.
# def Slf_trainer():

# backbone = resnet50(pretrained= True)    
# backbone = nn.Sequential(*list(backbone.children())[:-1]).to(device)
# projection_head = Projection_class(2048, 1024, 128).to(device)
# model = SimCLR(backbone, projection_head)
# optimizer = optim.SGD(model.parameters(), lr=0.001)
# criterion = NTXentLoss()
# transform = transforms.Compose([transforms.ToTensor(),
#                                 transforms.Resize((255, 255)),
#                                ])
# train_set = datasets.ImageFolder(root='C:/Users/abassoma/Documents/Dataset/Cassava/for_training',transform=transform)
# val_set = datasets.ImageFolder(root='C:/Users/abassoma/Documents/Dataset/Cassava/for_training',transform=transform)
# batch_size = 32
# train_loader = torch.utils.data.DataLoader(train_set,batch_size,shuffle=True)
# test_loader = torch.utils.data.DataLoader(val_set,batch_size,shuffle=False)
# Simclr_trainer(model=model, train_loader=train_loader, criterion=criterion, optimizer=optimizer, epochs=10, active_groups=["rotations"])