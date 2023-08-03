from pathlib import Path
from PIL import Image

if __name__ == '__main__':
    
    # files = [p for p in Path('data/trainset').glob('**/*.jpg')]
    
    # max_w = 0
    # max_h = 0
    # w_name = ''
    # h_name = ''
    # for file in files:
    #     img = Image.open(file)
    #     width, height = img.size
    #     print(file.name)
    #     if width > max_w:
    #         w_name = file.name
    #         max_w = width
    #     if height > max_h:
    #         h_name = file.name 
    #         max_h = height           
        
    # print(f"max width: {max_w} \n max height: {max_h} \n max width file name: {w_name} \n max height name: {h_name}")
    
    dic = {}
    with open('data/trainset/train.txt', 'r') as f:
        for raw in f:
            name = raw.split('.jpg')[0].strip()
            label = raw.split('.jpg')[-1].strip()
            #print(f"name: {name} label: {label}")
            dic[name] = label
    
    name = "A*otDrR42uc_YAAAAAAAAAAAAAAQAAAQ"
    label = dic[name]        
    print(f"name: {name} label: {label}")
    #print(dic)

        
            