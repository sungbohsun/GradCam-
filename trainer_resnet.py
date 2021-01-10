import os
import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim

from glob import glob
from PIL import Image
from torchvision import transforms
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


class Resnet_re(nn.Module):
    
    def __init__(self):
        super(Resnet_re, self).__init__()
        self.model   = models.resnet50(pretrained=True)
        self.conv1   = self.model.conv1 
        self.bn1     = self.model.bn1 
        self.relu    = self.model.relu 
        self.maxpool = self.model.maxpool 
        self.layer1  = self.model.layer1 
        self.layer2  = self.model.layer2
        self.layer3  = self.model.layer3
        self.layer4  = self.model.layer4
        self.avgpool = self.model.avgpool        
        self.linear  =  nn.Sequential(
                        nn.Linear(2048, 512),
                        nn.Dropout(0.2),
                        nn.ReLU(inplace=True),
                        nn.Linear(512,1))

    def forward(self, x):
                        
        for param in self.parameters():
            param.requires_grad = True
        
        for param in self.linear.parameters():
            param.requires_grad = True
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(64,-1)
        x = self.linear(x)
        
        return x
    
print('use',torch.cuda.get_device_name(0))

# check GPU availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = 'cpu'
print(device)

if not os.path.isdir('model/resnet'):
    os.mkdir('model/resnet')

if not os.path.isfile('tensor_dict.pt'):
    xs = torch.LongTensor([])
    ys = torch.LongTensor([])
    files = glob('beauty_re/*')
    for file in tqdm(files):
        input_image = Image.open(file)
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        x = preprocess(input_image)
        y = torch.tensor(int(file.split('/')[1].split('_')[2].split('.')[0]))
        xs = torch.cat((xs,x.unsqueeze(0)))
        ys = torch.cat((ys,y.unsqueeze(0)))
    ys = ys/100
    
    #save
    dic = {'xs':xs,'ys':ys}
    torch.save(dic, 'tensor_dict.pt')

else:
    dic = torch.load('tensor_dict.pt')
    xs,ys = [dic[c] for c in dic]
    

train_index,test_index = train_test_split(list(range(len(ys))),test_size=0.2, random_state=42)
X_train = xs[train_index]
y_train = ys[train_index]
X_test = xs[test_index]
y_test = ys[test_index]
dataloader_X_train = DataLoader(X_train,batch_size=64,shuffle=False, num_workers=0,drop_last=True)
dataloader_y_train= DataLoader(y_train,batch_size=64,shuffle=False, num_workers=0,drop_last=True)
dataloader_X_test = DataLoader(X_test,batch_size=64,shuffle=False, num_workers=0,drop_last=True)
dataloader_y_test= DataLoader(y_test,batch_size=64,shuffle=False, num_workers=0,drop_last=True)

model = Resnet_re().to(device)
loss_fn = nn.MSELoss()
learning_rate = 2e-5
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
er = 0
epoch = 0
best_loss = 100000
train_loss,valid_loss = [],[]
while er <=50 :
    epoch += 1
    loss_tmp = 0
    for batch_num,(x,y) in enumerate(zip(dataloader_X_train,dataloader_y_train)):
        model.train()
        output = model(x.to(device))
        loss = loss_fn(output, y.to(device))
        if torch.flatten(torch.isnan(loss)).any():
            continue
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_tmp += loss
    train_loss += [loss_tmp/batch_num]

    print('epoch',epoch,'train_loss:',round(float(loss_tmp/batch_num),6),end=' ')
    
    loss_tmp_valid = 0
    for batch_num,(x,y) in enumerate(zip(dataloader_X_test,dataloader_y_test)):
        model.eval()
        with torch.no_grad():
            output = model(x.to(device))
            loss = loss_fn(output, y.to(device))
            loss_tmp_valid += loss
    valid_loss += [loss_tmp_valid/batch_num]

    
    print('valid_loss:',round(float(loss_tmp_valid/batch_num),6))
    
    if loss_tmp_valid < best_loss: 
        best_loss = loss_tmp_valid
        torch.save(model.state_dict(), 'model/resnet/best_model_unfreeze'+str(epoch)+'.pt')
        er = 0
    else : 
        er += 1
print('stop at epoch:',epoch)
dic = {'train_loss':train_loss,'valid_loss':valid_loss}
torch.save(dic, 'loss_resnet_unfreeze.pt')