from profill_loader import get_loader
from PIL import Image
from etaprogress.progress import ProgressBar
import os
import requests
import time
import matplotlib.pyplot as plt

if __name__=="__main__":
    if not os.path.exists('baselines/profill/results/'):
            os.makedirs('baselines/profill/results/')
    
    if not os.path.exists('baselines/profill/imgs/'):
            os.makedirs('baselines/profill/imgs/')
    
    if not os.path.exists('baselines/profill/masks/'):
            os.makedirs('baselines/profill/masks/')

    coco_test_loader = get_loader(root='datasets/coco/val2017', \
                                        json='datasets/coco/annotations/instances_val2017.json', \
                                            shuffle=True)

    total = len(coco_test_loader)
    bar = ProgressBar(total, max_width=80)
    for i, data in enumerate(coco_test_loader, 0):
        bar.numerator = i+1
        print(bar, end='\r')
        images, masks, paths = data
        mask = masks[0]
        image = images[0]
        path = paths[0]

        image = image.permute(1, 2, 0).numpy()
        mask = mask.squeeze(0).numpy() * 255.
        
        image = image.astype('uint8')
        mask = mask.astype('uint8')

        image = Image.fromarray(image,'RGB')
        mask = Image.fromarray(mask,'L')

        image.save("baselines/profill/imgs/{}.jpg".format(i))
        mask.save("baselines/profill/masks/{}.jpg".format(i))
        
        mode_img = image.mode
        mode_msk = mask.mode

        W, H = image.size
        str_img = image.tobytes().decode("latin1")
        str_msk = mask.tobytes().decode("latin1")

        data = {'str_img': str_img, 'str_msk': str_msk, 'width':W, 'height':H, 
                'mode_img':mode_img, 'mode_msk':mode_msk,  'is_refine': True}
        
        time.sleep(0.01)
        r = requests.post('http://47.57.135.203:2333/api', json=data)
        str_result = r.json()['str_result']
        result = str_result.encode("latin1")
        result = Image.frombytes('RGB', (W, H), result, 'raw')
        result.save("baselines/profill/results/{}".format(i))