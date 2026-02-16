from augmentation import *
from tqdm import tqdm
import matplotlib.pyplot as plt
from contrastive_approaches import *
from torchvision.models import resnet50
import torch.optim as optim
import lightly
import matplotlib.pyplot as plt
from lightly.loss import NTXentLoss
import torchvision.datasets as datasets

from datamodule import CassavaDataModule
from utils import *
from pytorch_metric_learning.losses import SupConLoss
device = ("cuda" if torch.cuda.is_available() else "cpu")



# Train supcon with SupCon losses
def supcon_trainer(model, train_loader, criterion, optimizer, epochs, active_groups):
    epoch_losses = []
    running_loss = 0
    for step, (data, label) in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()
        first_batch, second_batch = global_global_augmentation(data, active_groups, device)
        batch_label = label.shape[0]
        instances_batch  = torch.cat([first_batch, second_batch], dim=0)
        z_instance_batch = model(instances_batch)
        z_first_batch, z_second_batch = torch.split(z_instance_batch, [batch_label, batch_label], dim=0)
        z = torch.cat([z_first_batch, z_second_batch], dim=0)
        labels = torch.cat([label, label], dim=0)
        loss = criterion(z, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    epoch_loss = running_loss / len(train_loader)
    epoch_losses.append(epoch_loss)
    print(epoch_losses)
    plot_training_loss(epoch_loss, epochs)
          
# Train Simclr with the NT-Xent losses

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
  # plot_training_loss(epoch_loss, epochs)
        
# Train Slfpn with the MSE loss
        
def Slfpn_trainer(model, projector, train_loader, criterion, optimizer, epochs, active_groups):
    epoch_losses = []
  
    for epoch in range(epochs):
        running_loss = 0.0
        print(f"Début de l'époque {epoch + 1}")
        for step, (data, label) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            batch_1, batch_2 = global_global_augmentation(data, active_groups)
            batch_1, batch_2 = batch_1.to(device), batch_2.to(device)
            data = data.to(device)
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
    # plot_training_loss(epoch_loss, epochs)
    

# # train Slf with an MSE loss.
# def Slf_trainer():

backbone = resnet50(pretrained=True)
backbone = nn.Sequential(*list(backbone.children())[:-1]).to(device)
projection_head = Projection_class(2048, 1024, 128).to(device)
model = SimCLR(backbone, projection_head)
optimizer = optim.SGD(model.parameters(), lr=0.001)
# criterion_simclr = NTXentLoss()
criterion = SupConLoss()
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize((255, 255)),
                               ])
# train_set = datasets.ImageFolder(root='C:/Users/abassoma/Documents/Dataset/Cassava/for_training',transform=transform)
# val_set = datasets.ImageFolder(root='C:/Users/abassoma/Documents/Dataset/Cassava/for_training',transform=transform)
batch_size = 32


img_dir ='C:/Users/rkonan/Desktop/Deep_Learning_Agriculture/Dataset/Kaggle_cassava-leaf-disease-classification/small_toy/images'
csv_file ='C:/Users/rkonan/Desktop/Deep_Learning_Agriculture/Dataset/Kaggle_cassava-leaf-disease-classification/small_toy/train.csv'
json_file='C:/Users/rkonan/Desktop/Deep_Learning_Agriculture/Dataset/Kaggle_cassava-leaf-disease-classification/small_toy/label_num_to_disease_map.json'

dm = CassavaDataModule(data_dir=(img_dir, csv_file, json_file), batch_size = batch_size, input_img_size = (224, 224),
                 val_split = 0.2, test_split = 0.2, n_transforms_to_choose=2,
                 base_transform_to_use= 'resnet50',
                 num_workers=0, random_seed=42)
dm.setup(stage="fit") # calling in the fit stage to initialize the train and validation dataloaders
train_set= dm.train_dataloader()
val_set= dm.val_dataloader()
train_loader = dm.train_dataloader()
test_loader = dm.val_dataloader()

Simclr_trainer(model=model, train_loader=train_loader, criterion=criterion, optimizer=optimizer, epochs=3, active_groups=["rotations"])
