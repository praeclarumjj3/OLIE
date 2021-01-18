from datasets.gen_masked_image_loader import get_loader
from PIL import Image
from etaprogress.progress import ProgressBar
import os
import requests

"""
Image Inpainting done using:
ProFill: High-Resolution Image Inpainting with Iterative Confidence Feedback and Guided Upsampling, ECCV 2020
"""

if __name__=="__main__":
    if not os.path.exists('datasets/coco/inpainted_test/'):
            os.makedirs('datasets/coco/inpainted_test/')
    

#     coco_train_loader = get_loader(device=device, \
#                                     root='coco/train2017', \
#                                         json='coco/annotations/instances_train2017.json', \
#                                             batch_size=1, \
#                                                 shuffle=False)

    coco_test_loader = get_loader(root='datasets/coco/val2017', \
                                        json='datasets/coco/annotations/instances_val2017.json', \
                                            batch_size=1, \
                                                shuffle=True)

    total = len(coco_test_loader)
    bar = ProgressBar(total, max_width=80)
    for i, data in enumerate(coco_test_loader, 0):
        bar.numerator = i+1
        print(bar, end='\r')
        images, masks, paths = data
        image = images[0]
        mask = masks[0]
        path = paths[0]
        mode_img = image.mode
        mode_msk = mask.mode

        W, H = image.size
        str_img = image.tobytes().decode("latin1")
        str_msk = mask.tobytes().decode("latin1")

        data = {'str_img': str_img, 'str_msk': str_msk, 'width':W, 'height':H, 
                'mode_img':mode_img, 'mode_msk':mode_msk}

        r = requests.post('http://47.57.135.203:2333/api', json=data)
        str_result = r.json()['str_result']
        result = str_result.encode("latin1")
        result = Image.frombytes('RGB', (W, H), result, 'raw')
        result.save("datasets/coco/inpainted_test/{}".format(path))