"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from modules.networks.sync_batchnorm import DataParallelWithCallback
from modules.olie_gan import OlieGAN
from modules.helpers.utils import tensor_to_list

class OlieTrainer():
    """
    Trainer creates the model and optimizers, and uses them to
    updates the weights of the network while reporting losses
    and the latest visuals to visualize the progress in training.
    """

    def __init__(self, opt, solo):
        self.opt = opt
        self.olie_model = OlieGAN(opt)
        self.solo = solo

        if len(opt.gpu_ids) > 0:
            self.olie_model = DataParallelWithCallback(self.olie_model,
                                                          device_ids=opt.gpu_ids)
            self.olie_model.cuda()
            self.olie_model_on_one_gpu = self.olie_model.module
            
            self.solo = DataParallelWithCallback(self.solo,
                                                          device_ids=opt.gpu_ids)
            self.solo.cuda()
            self.solo_on_one_gpu = self.solo.module
        else:
            self.olie_model_on_one_gpu = self.olie_model
            self.solo_on_one_gpu = self.solo

        self.generated = None

        if opt.isTrain:
            self.optimizer_G, self.optimizer_D = \
                self.olie_model_on_one_gpu.create_optimizers(opt)
            self.old_lr = opt.lr

        # print(self.olie_model_on_one_gpu.netG)
        # print(self.olie_model_on_one_gpu.netD)

    def run_generator_one_step(self, data):
        imgs, _ = data
        solo_imgs = tensor_to_list(imgs)
        self.optimizer_G.zero_grad()
        maps = self.solo(solo_imgs)
        g_losses, generated, masked, semantics = self.olie_model(data, mode='generator')
        g_loss = sum(g_losses.values()).mean()
        g_loss.backward()
        self.optimizer_G.step()
        self.g_losses = g_losses
        self.generated = generated
        self.masked = masked
        self.semantics = semantics

    def run_discriminator_one_step(self, data):
        self.optimizer_D.zero_grad()
        d_losses = self.olie_model(data, mode='discriminator')
        d_loss = sum(d_losses.values()).mean()
        d_loss.backward()
        self.optimizer_D.step()
        self.d_losses = d_losses

    def get_latest_losses(self):
        return {**self.g_losses, **self.d_losses}

    def get_latest_generated(self):
        return self.generated

    def get_latest_real(self):
        return self.olie_model_on_one_gpu.real_shape

    def get_semantics(self):
        return self.semantics

    def get_mask(self):
        if self.masked.shape[1] == 3:
            return self.masked
        else:
            return self.masked[:,:3]

    def update_learning_rate(self, epoch):
        self.update_learning_rate(epoch)

    def save(self, epoch):
        self.olie_model_on_one_gpu.save(epoch)

    ##################################################################
    # Helper functions
    ##################################################################

    def update_learning_rate(self, epoch):
        if epoch > self.opt.niter:
            lrd = self.opt.lr / self.opt.niter_decay
            new_lr = self.old_lr - lrd
        else:
            new_lr = self.old_lr

        if new_lr != self.old_lr:
            if self.opt.no_TTUR:
                new_lr_G = new_lr
                new_lr_D = new_lr
            else:
                new_lr_G = new_lr / 2
                new_lr_D = new_lr * 2

            for param_group in self.optimizer_D.param_groups:
                param_group['lr'] = new_lr_D
            for param_group in self.optimizer_G.param_groups:
                param_group['lr'] = new_lr_G
            print('update learning rate: %f -> %f' % (self.old_lr, new_lr))
            self.old_lr = new_lr
