#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch as t
from torch.utils import data


# In[2]:


import os
from PIL import Image
import numpy as np
class DogCat(data.Dataset):
    def __init__(self,root):
        imgs=os.listdir(root)
        self.imgs=[os.path.join(root,img) for img in imgs]
    def __getitem__(self,index):
        img_path=self.imgs[index]
        label=1 if 'dog' in img_path.split('/')[-1] else 0
        pil_img=Image.open(img_path)
        array=np.asarray(pil_img)
        data=t.from_numpy(array)
        return data,label
    def __len__(self):
        return len(self.imgs)


# In[3]:


dataset=DogCat('/home/wcj/pytorch-book/chapter5-常用工具/data/dogcat')
img,label=dataset[0]
for img,label in dataset:
    print(img.size(),img.float().mean(),label)
    


# In[4]:


img.shape


# In[5]:


import os
from PIL import Image
import numpy as np
from torchvision import transforms as T
transforms=T.Compose([
    T.Resize(224),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[.5,.5,.5],std=[.5,.5,.5])
])
class DogCat(data.Dataset):
    def __init__(self,root,transforms=None):
        imgs=os.listdir(root)
        self.imgs=[os.path.join(root,img) for img in imgs]
        self.transforms=transforms
    def __getitem__(self,index):
        img_path=self.imgs[index]
        label=0 if 'dog' in img_path.split('/')[-1] else 1
        data=Image.open(img_path)
        if self.transforms:
            data=self.transforms(data)
        return data,label
    def __len__(self):
        return len(self.imgs)
dataset=DogCat('/home/wcj/pytorch-book/chapter5-常用工具/data/dogcat',transforms=transforms)
img,label=dataset[0]
for img,label in dataset:
    print(img.size(),label)


# In[6]:


get_ipython().run_line_magic('help', 'T.CenterCrop')


# In[7]:


from torchvision.datasets import ImageFolder
dataset=ImageFolder('/home/wcj/pytorch-book/chapter5-常用工具/data/dogcat_2')


# In[8]:


dataset.class_to_idx


# In[9]:


dataset.imgs


# In[10]:


dataset[0][0]


# In[11]:


dataset[1][0]


# In[12]:


normalize=T.Normalize(mean=[0.4,0.4,0.4],std=[0.2,0.2,0.2])
transform=T.Compose([
    T.RandomResizedCrop(224),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    normalize,
])


# In[13]:


dataset=ImageFolder('/home/wcj/pytorch-book/chapter5-常用工具/data/dogcat_2',transform=transform)


# In[14]:


dataset[0][0].size()


# In[15]:


to_img=T.ToPILImage()


# In[16]:


to_img(dataset[0][0]*0.2+0.4)


# In[17]:


from torch.utils.data import DataLoader
dataloader=DataLoader(dataset,batch_size=3,shuffle=True,num_workers=0,drop_last=False)
dataiter=iter(dataloader)
imgs,labels=next(dataiter)
imgs.size()


# batch_size,channel,height,weight

# In[18]:


class NewDogCat(DogCat):
    def __getitem__(self,index):
        try:
            return super(NewDogCat,self).__getitem__(index)
        except:
            return None,None
from torch.utils.data.dataloader import default_collate
import ipdb
# ipdb.set_trace()
def my_collate_fn(batch):
    batch=list(filter(lambda x:x[0] is not None,batch))
    if len(batch)==0: return t.Tensor()
    return default_collate(batch)
dataset=NewDogCat('/home/wcj/pytorch-book/chapter5-常用工具/data/dogcat_wrong/',transforms=transform)
    


# In[19]:


dataset[2]


# In[21]:


dataloader=DataLoader(dataset,2,collate_fn=my_collate_fn,num_workers=1,shuffle=True)
for batch_datas,batch_labels in dataloader:
    print(batch_datas.size(),batch_labels.size())


# In[25]:


dataset=DogCat('/home/wcj/pytorch-book/chapter5-常用工具/data/dogcat/',transforms=transform)
weigths=[2 if label==1 else 1 for data,label in dataset]
weigths


# In[28]:


from torch.utils.data.sampler import WeightedRandomSampler
sampler=WeightedRandomSampler(weigths,num_samples=9,replacement=True)
dataloader=DataLoader(dataset,batch_size=3,sampler=sampler)
for datas,labels in dataloader:
    print(labels.tolist())


# In[30]:


sampler=WeightedRandomSampler(weigths,8,replacement=False)
dataloader=DataLoader(dataset,batch_size=4,sampler=sampler)
for datas,labels in dataloader:
    print(labels.tolist())


# In[31]:


from torchvision import models
from torch import nn
resnet34=models.squeezenet1_1(pretrained=True,num_classes=1000)
resnet34.fc=nn.Linear(512,10)
from torchvision import datasets
dataset=datasets.MNIST('data/',download=True,train=False,transform=transform)


# In[32]:


from torchvision import transforms
to_pil=transforms.ToPILImage()
to_pil(t.randn(3,64,64))


# In[33]:


len(dataset)


# In[39]:


dataloader=DataLoader(dataset,shuffle=True,batch_size=16)
from torchvision.utils import make_grid,save_image
dataiter=iter(dataloader)
img=make_grid(next(dataiter)[0],4)
to_img(img)


# In[41]:


save_image(img,'a.png')


# In[42]:


Image.open('a.png')


# In[44]:


import torch as t
import visdom
vis=visdom.Visdom(env=u'test1',use_incoming_socket=False)
x=t.arange(1,30,0.01)
y=t.sin(x)
vis.line(X=x,Y=y,win='six',opts={'title':'y=sin(x)'})


# In[45]:


for ii in range(0,10):
    x=t.Tensor([ii])
    y=x
    vis.line(X=x,Y=y,win='polynomial',update='append' if ii>0 else None)
x=t.arange(0,9,0.1)
y=(x**2)/9
vis.line(X=x,Y=y,win='polynomial',name='this is a new Trace',update='new')


# In[50]:


vis.image(t.randn(64,64).numpy())
vis.image(t.randn(3,64,64).numpy(),win='random2')
vis.images(t.randn(36,3,64,64).numpy(),nrow=6,win='random3',opts={'title':'random_imgs'})


# In[51]:


vis.text(u'''<h1>hello visdom</h1><br>visdom''',win='visdom',opts={'title':u'visdom简介'})


# In[4]:


import torch as t
inwav=t.randn(1,1,128)
inwav


# In[5]:


inwav.shape


# In[8]:


inwav.shape[0]


# In[10]:


inwav[0,0,0:10]


# In[11]:


idx=t.zeros(1)


# In[12]:


idx


# In[15]:


import numpy as np
idy=np.zeros((1,1,1,128))


# In[16]:


idy


# In[22]:


idy.shape[0]


# In[25]:





# In[26]:


conda list


# In[27]:


pip list


# In[1]:


import torch as t


# In[2]:


tensor=t.Tensor(3,4)


# In[3]:


tensor


# In[4]:


tensor.cuda(0)


# In[6]:


tensor.is_cuda


# In[7]:


tensor=tensor.cuda()


# In[8]:


tensor.is_cuda


# In[9]:


from torch import nn


# In[10]:


module=nn.Linear(3,4)


# In[11]:


module.cuda(device=1)


# In[12]:


module.weight


# In[16]:


module.bias


# In[17]:


module.bias.shape


# In[133]:


ceriterion=t.nn.CrossEntropyLoss(weight=t.Tensor(1,4))


# In[142]:


input=t.randn(1,4,4).cuda()
input.shape


# In[135]:


target=t.Tensor([[1,0,0,1]]).long().cuda()
target.shape


# In[136]:


# loss=ceriterion(input,target)
ceriterion.cuda()


# In[138]:


loss=ceriterion(input,target)


# In[139]:


loss


# In[141]:


loss


# In[143]:


ceriterion._buffers


# In[145]:


x=t.cuda.FloatTensor(2,3)
y=t.FloatTensor(2,3).cuda()
with t.cuda.device(1):
    a=t.cuda.FloatTensor(2,3)
    b=t.FloatTensor(2,3).cuda()
    print(a.get_device()==b.get_device()==1)
    c=a+b
    print(c.get_device()==1)
    z=x+y
    print(z.get_device()==0)
    d=t.randn(2,3).cuda(0)
    print(d.get_device()==2)
    


# In[147]:


t.set_default_tensor_type('torch.cuda.FloatTensor')
a=t.ones(2,3)
a.is_cuda


# In[149]:


a=t.Tensor(3,4)
if t.cuda.is_available():
    a=a.cuda(1)
    t.save(a,'a.pth')
    b=t.load('a.pth')
    c=t.load('a.pth',map_location=lambda storage,loc:storage )
    d=t.load('a.pth',map_location={'cuda:1':'cuda:0'})


# In[150]:


t.set_default_tensor_type('torch.FloatTensor')
from torchvision.models import SqueezeNet
model=SqueezeNet()
model.state_dict().keys()


# In[151]:


t.save(model.state_dict(),'squeezenet.pth')
model.load_state_dict(t.load('squeezenet.pth'))


# In[152]:


optimizer=t.optim.Adam(model.parameters(),lr=0.01)


# In[153]:


t.save(optimizer.state_dict(),'optimizer.pth')
optimizer.load_state_dict(t.load('optimizer.pth'))


# In[154]:


all_data=dict(optimizer=optimizer.state_dict(),model=model.state_dict(),info=u'模型和优化器的所有参数')
t.save(all_data,'all.pth')


# In[155]:


all_data=t.load('all.pth')
all_data.keys()


# In[ ]:




