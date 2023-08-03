import torch
import argparse
from torchvision import Transforms as T
from PIL import Image
from model import Unet

parser = argparse.ArgumentParser()
parser.add_argument('picture_path')

args = parser.parse_args()

if __name__ == '__main__':
    
    model = Unet(16, dim_mults=(1, 2, 4, 8))
    
    model.load_state_dict(torch.load('results/model.pth'))
    
    #图像处理
    
    img = Image.open(args.picture_path)
    
    img2tensor = T.Compose([T.Resize(256),
                            T.ToTensor()]) 
    
    tensor = img2tensor(img)
    
    model.eval()
    pred = model(tensor)
    
    print(f"预测年龄: {pred}")