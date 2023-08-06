import torch
import argparse
from torchvision import transforms as T
from PIL import Image
from model import Unet, my_device

parser = argparse.ArgumentParser()
parser.add_argument('picture_path')

args = parser.parse_args()

if __name__ == '__main__':
    
    p_model = Unet(16, dim_mults=(1, 2, 4, 8))
    
    p_model.load_state_dict(torch.load('results/model.pth', map_location=torch.device('mps')))
    
    #图像处理
    print(args.picture_path)
    img = Image.open(args.picture_path)
    
    img2tensor = T.Compose([T.Resize(size=(256, 256)),
                            T.ToTensor()]) 
    
    tensor = img2tensor(img)
    tensor = torch.unsqueeze(tensor, 0)
    print(f"device: {my_device}")
    tensor = tensor.to(my_device)
    
    p_model.eval()
    pred = p_model(tensor).item()
    
    print(f"预测年龄: {pred}")