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
from torchvision.models import vgg19

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        
        # get the pretrained VGG19 network
        self.vgg = vgg19(pretrained=True)
        
        # disect the network to access its last convolutional layer
        self.features_conv = self.vgg.features[:36]
        
        # get the max pool of the features stem
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        
        # get the classifier of the vgg19
        self.classifier = self.vgg.classifier[:5]
        # change the number of classes 
        self.classifier2 = nn.Linear(4096, 512)
        self.classifier3 = nn.Linear(512, 1)
        
    def forward(self, x):
        
        for param in self.parameters():
            param.requires_grad = True
        
        for param in self.classifier.parameters():
            param.requires_grad = True
            
        for param in self.classifier2.parameters():
            param.requires_grad = True
            
        for param in self.classifier3.parameters():
            param.requires_grad = True
            
        x = self.features_conv(x)
        x = self.max_pool(x)
        x = x.view((64,-1))
        x = self.classifier(x)
        x = self.classifier2(x)
        x = self.classifier3(x)
        return x

print('use',torch.cuda.get_device_name(0))

# check GPU availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = 'cpu'
print(device)

if not os.path.isdir('model/vgg'):
    os.makedirs('model/vgg')

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

model = VGG().to(device)
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
        torch.save(model.state_dict(), 'model/vgg/best_model_unfreeze'+str(epoch)+'.pt')
        er = 0
    else : 
        er += 1
print('stop at epoch:',epoch)
dic = {'train_loss':train_loss,'valid_loss':valid_loss}
torch.save(dic, 'loss_vgg_unfreeze.pt')