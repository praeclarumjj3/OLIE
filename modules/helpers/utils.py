import torch
import matplotlib.pyplot as plt
import random
from torchvision import utils
import numpy as np
import os
from PIL import Image
from modules.helpers import coco
import importlib

def tensor_to_list(maps):
    
    masks = []
    maps = maps.squeeze(0)

    for i in range(maps.shape[0]):
        masks.append(maps[i])

    return masks

def masking(image, phase, index):
    if phase=="single":
        # print(torch.mean(image[i][index]))
        for i in range(int(image.shape[0])):
            image[i][index] = torch.tensor(0, dtype = float)
    elif phase=="multi":
        for i in range(image.shape[0]):
            # print(torch.mean(image[i][index[0]:index[1]]))
            image[i][index[0]:index[1]] = torch.tensor(0, dtype = float)
    
    return image

def masking_objects(image, phase, c_index, x, w, y, h):
    if phase=="single":
        # print(torch.mean(image[i][c_index,y:y+h,x:x+w]))
        for i in range(int(image.shape[0])):
            image[i][c_index,y:y+h,x:x+w] = torch.tensor(0, dtype = float)
    elif phase=="multi":
        for i in range(image.shape[0]):
            # print(torch.mean(image[i][c_index[0]:c_index[1],y:y+h,x:x+w]))
            image[i][c_index[0]:c_index[1],y:y+h,x:x+w] = torch.tensor(0, dtype = float)
    
    return image

def masking_threshold(threshold, image, phase, c_index, x, w, y, h):
    overlap = torch.ones_like(image)
    if phase=="single":
        for i in range(int(overlap.shape[0])):
            # print(torch.mean(image[i][c_index,y:y+h,x:x+w]))
            overlap[i][c_index,y:y+h,x:x+w] = (image[i][c_index,y:y+h,x:x+w] > threshold).float() 
    elif phase=="multi":
        for i in range(overlap.shape[0]):
            # print(torch.mean(image[i][c_index[0]:c_index[1],y:y+h,x:x+w]))
            overlap[i][c_index[0]:c_index[1],y:y+h,x:x+w] = (image[i][c_index[0]:c_index[1],y:y+h,x:x+w] > threshold).float()
    
    return image*overlap

def normalize(inputs):
    pixel_mean = torch.Tensor([103.530, 116.280, 123.675]).view(3, 1, 1).cuda()
    pixel_std = torch.Tensor([57.375, 57.120, 58.395]).view(3, 1, 1).cuda()
    normalizer = lambda x: (x.cuda() - pixel_mean) / pixel_std
    return normalizer(inputs)

def mask_shuffle(image, index, phase):

    masks_start = []
    masks = []
    image = image.squeeze(0)

    for i in range(index[0]):
        masks_start.append(image[i])

    for i in range(index[0],index[1]):
        masks.append(image[i])
    
    if phase == "reverse":
        masks.reverse()
    elif phase == "random":
        random.shuffle(masks)

    masks = masks_start + masks

    for i in range(index[1],image.shape[0]):
        masks.append(image[i])

    masks = torch.stack(masks, dim=0)

    return masks.unsqueeze(0)

def visualize(x,layer):
    plt.rcParams.update({'font.size': 3})
    dim = int(x.shape[1])
    x = x[0].cpu() 
    x = x.permute(1, 2, 0).numpy()
    f, axarr = plt.subplots(int(dim**0.5),int(dim**0.5),figsize=(16,16))
    for j in range(int(dim**0.5)*int(dim**0.5)):
        r = int(j/int(dim**0.5))
        c = int(j%int(dim**0.5))
        axarr[r,c].imshow(x[:,:,j])
        axarr[r,c].axis('off')
    f.savefig('visualizations/{}.jpg'.format(layer))


def visTensor(tensor, name, ch=0, allkernels=False, nrow=8, padding=1): 
    
    n,c,w,h = tensor.shape

    if allkernels: 
        tensor = tensor.view(n*c, -1, w, h)
    elif c != 3: 
        tensor = tensor[:,ch,:,:].unsqueeze(dim=1)
        
    rows = np.min((tensor.shape[0] // nrow + 1, 64))    
    grid = utils.make_grid(tensor, nrow=nrow, normalize=True, padding=padding)
    
    plt.figure(figsize=(nrow,rows))
    plt.axis('off')
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    
    plt.savefig('visualizations/{}.jpg'.format(name))
    
def visualize_kernels(model, name):
    if len(name) == 0:
        return

    filters = getattr(model,name)
    filters = filters.weight.data.clone()

    visTensor(filters, name, ch=0, allkernels=False)

    ##########################################################################

def find_class_in_module(target_cls_name, module):
    target_cls_name = target_cls_name.replace('_', '').lower()
    clslib = importlib.import_module(module)
    cls = None
    for name, clsobj in clslib.__dict__.items():
        if name.lower() == target_cls_name:
            cls = clsobj

    if cls is None:
        print("In %s, there should be a class whose name matches %s in lowercase without underscore(_)" % (module, target_cls_name))
        exit(0)

    return cls

def save_image(image_numpy, image_path, create_dir=False):
    if create_dir:
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
    if len(image_numpy.shape) == 2:
        image_numpy = np.expand_dims(image_numpy, axis=2)
    if image_numpy.shape[2] == 1:
        image_numpy = np.repeat(image_numpy, 3, 2)
    image_pil = Image.fromarray(image_numpy)

    # save to png
    image_pil.save(image_path.replace('.jpg', '.png'))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def atoi(text):
    return int(text) if text.isdigit() else text


def save_network(net, label, epoch, opt):
    save_filename = '%s_net_%s.pth' % (epoch, label)
    save_path = os.path.join(opt.checkpoints_dir, opt.name, save_filename)
    torch.save(net.cpu().state_dict(), save_path)
    if len(opt.gpu_ids) and torch.cuda.is_available():
        net.cuda()


def load_network(net, label, epoch, opt):
    save_filename = '%s_net_%s.pth' % (epoch, label)
    save_dir = os.path.join(opt.checkpoints_dir, opt.name)
    save_path = os.path.join(save_dir, save_filename)
    weights = torch.load(save_path)
    net.load_state_dict(weights)
    return net

def tile_images(imgs, picturesPerRow=4):
    """ Code borrowed from
    https://stackoverflow.com/questions/26521365/cleanly-tile-numpy-array-of-images-stored-in-a-flattened-1d-format/26521997
    """

    # Padding
    if imgs.shape[0] % picturesPerRow == 0:
        rowPadding = 0
    else:
        rowPadding = picturesPerRow - imgs.shape[0] % picturesPerRow
    if rowPadding > 0:
        imgs = np.concatenate([imgs, np.zeros((rowPadding, *imgs.shape[1:]), dtype=imgs.dtype)], axis=0)

    # Tiling Loop (The conditionals are not necessary anymore)
    tiled = []
    for i in range(0, imgs.shape[0], picturesPerRow):
        tiled.append(np.concatenate([imgs[j] for j in range(i, i + picturesPerRow)], axis=1))

    tiled = np.concatenate(tiled, axis=0)
    return tiled

# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8, normalize=True, tile=False):
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2im(image_tensor[i], imtype, normalize))
        return image_numpy

    if image_tensor.dim() == 4:
        # transform each image in the batch
        images_np = []
        for b in range(image_tensor.size(0)):
            one_image = image_tensor[b]
            one_image_np = tensor2im(one_image)
            images_np.append(one_image_np.reshape(1, *one_image_np.shape))
        images_np = np.concatenate(images_np, axis=0)
        if tile:
            images_tiled = tile_images(images_np)
            return images_tiled
        else:
            return images_np

    if image_tensor.dim() == 2:
        image_tensor = image_tensor.unsqueeze(0)
    image_numpy = image_tensor.detach().cpu().float().numpy()
    if normalize:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
    image_numpy = np.clip(image_numpy, 0, 255)
    if image_numpy.shape[2] == 1:
        image_numpy = image_numpy[:, :, 0]
    return image_numpy.astype(imtype)

# Converts a one-hot tensor into a colorful label map
def tensor2label(label_tensor, n_label, imtype=np.uint8, tile=False):
    if label_tensor.dim() == 4:
        # transform each image in the batch
        images_np = []
        for b in range(label_tensor.size(0)):
            one_image = label_tensor[b]
            one_image_np = tensor2label(one_image, n_label, imtype)
            images_np.append(one_image_np.reshape(1, *one_image_np.shape))
        images_np = np.concatenate(images_np, axis=0)
        if tile:
            images_tiled = tile_images(images_np)
            return images_tiled
        else:
            images_np = images_np[0]
            return images_np

    if label_tensor.dim() == 1:
        return np.zeros((64, 64, 3), dtype=np.uint8)
    if n_label == 0:
        return tensor2im(label_tensor, imtype)
    label_tensor = label_tensor.cpu().float()
    if label_tensor.size()[0] > 1:
        label_tensor = label_tensor.max(0, keepdim=True)[1]
    label_tensor = Colorize(n_label)(label_tensor)
    label_numpy = np.transpose(label_tensor.numpy(), (1, 2, 0))
    result = label_numpy.astype(imtype)
    return result

def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])


def labelcolormap(N):
    cmap = np.zeros((N, 3), dtype=np.uint8)
    for i in range(N):
        r, g, b = 0, 0, 0
        id = i + 1  # let's give 0 a color
        for j in range(7):
            str_id = uint82bin(id)
            r = r ^ (np.uint8(str_id[-1]) << (7 - j))
            g = g ^ (np.uint8(str_id[-2]) << (7 - j))
            b = b ^ (np.uint8(str_id[-3]) << (7 - j))
            id = id >> 3
        cmap[i, 0] = r
        cmap[i, 1] = g
        cmap[i, 2] = b

    if N == 182:  # COCO
        important_colors = {
            'sea': (54, 62, 167),
            'sky-other': (95, 219, 255),
            'tree': (140, 104, 47),
            'clouds': (170, 170, 170),
            'grass': (29, 195, 49)
        }
        for i in range(N):
            name = coco.id2label(i)
            if name in important_colors:
                color = important_colors[name]
                cmap[i] = np.array(list(color))

    return cmap

class Colorize(object):
    def __init__(self, n=35):
        self.cmap = labelcolormap(n)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.size()
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image[0]).cpu()
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image
