from pathlib import Path
from functools import partial
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch import nn
import torch
from torchvision import transforms as T, utils
from PIL import Image
from tqdm import tqdm
from model import Unet
from multiprocessing import cpu_count

class pet_dataset(Dataset):
    
    def __init__(self,
                 folder,
                 image_size,
                 clip_size=0,
                 set_type='train'):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.set_type = set_type
        
        self.paths = [p for p in Path(f"{folder}").glob("**/*.jpg")]
        self.labels = {}
        with open(folder + f'/{set_type}.txt') as f:
            for raw in f:
                name = raw.split('.jpg')[0].strip()
                label = raw.split('.jpg')[-1].strip()
                self.labels[name] = label

        if set_type == 'train':
            self.transform = T.Compose([T.Resize(image_size),
                                        T.RandomHorizontalFlip(),
                                        T.CenterCrop(image_size - clip_size),
                                        T.ToTensor()])
        else:
            self.transform = T.Compose([T.Resize(image_size),
                                        T.ToTensor()])
    
    
    def __len__(self):
        return len(self.paths)


    def __getitem__(self, index):
        path = self.paths[index]
        name = path.name.split('.jpg')[0].strip()
        img = Image.open(path)
        return self.transform(img), torch.tensor(int(self.labels[name]), dtype=torch.float32)

        

class trainer(object):
    
    # dataset initialize
    def __init__(self,
                model, 
                folder, 
                train_batch_size=16, 
                train_lr=1e-4,
                train_num_step=100000,
                adma_beta=(0.9, 0.99),
                result_folder='./results'):
        
        
        self.model = model
        self.channels = model.channels
        self.image_size = 256
        
        self.batch_size = train_batch_size
        self.train_num_steps = train_num_step
        
        ds = pet_dataset(folder + '/trainset', self.image_size)
        self.tr_dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=cpu_count())
        
        ds = pet_dataset(folder + '/valset', self.image_size, set_type='val')
        self.ts_dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=cpu_count())
        
        self.opt = Adam(model.parameters(), lr=train_lr, betas=adma_beta)
        
        self.results_folder = Path(result_folder)
        self.results_folder.mkdir(exist_ok=True)
        
        self.step = 0
        self.device = "mps"
    
    
    def train(self, step=0, load_model=False):
        
        if load_model:
            self.step = step
            self.load()
            
        self.model.train()
        
        with tqdm(initial = self.step, total = self.train_num_steps) as pbar:
            
            while self.step < self.train_num_steps:
                
                data, label = next(iter(self.tr_dl))
                #print(f"{data}, {label}")
                data = data.to(self.device)
                label = label.to(self.device)
                
                pred = self.model(data)
                
                loss = nn.functional.mse_loss(pred, label)
                
                loss.backward()
                
                self.opt.step()
                self.opt.zero_grad()
                
                self.step = self.step + 1
                
                loss = loss.item()
                print(f"loss: {loss}")
                pbar.update(1)
                if self.step % 100 == 0:
                    self.test(self.ts_dl)
                    self.save()
    
    
    def test(self, test_loader):
        
        loss_fun = nn.L1Loss(reduction='mean')
        
        self.model.eval()
        
        test_loss = 0
        with torch.no_grad():
            x, y = next(iter(test_loader))
            x, y = x.to(self.device), y.to(self.device)
            pred = self.model(x)
            test_loss = loss_fun(pred, y).item()
                     
        print(f"loss in test set: {test_loss}")
    
        
    def save(self):
        torch.save(self.model.state_dict(), 'results/model.pth')
        print("save model param.")
    
    
    def load(self):
        self.model.load_state_dict(torch.load('results/model.pth'))
        print("loading model param")

    

if __name__ == '__main__':
    
    p_net = Unet(8, dim_mults=(1, 2))
    
    t = trainer(p_net,
                'data'
                )
    
    t.train()
    
    
    
    
                