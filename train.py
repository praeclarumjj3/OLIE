"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved. Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import os
if "STY" not in os.environ.keys():
    import multiprocessing
    multiprocessing.set_start_method('spawn', True)

import sys
from collections import OrderedDict
from options.train_options import TrainOptions
from datasets.coco_loader import get_loader
import torch
from modules.helpers.iter_counter import IterationCounter
from modules.helpers.visualizer import Visualizer
from trainers.olie_trainer import OlieTrainer
from detectron2.checkpoint import DetectionCheckpointer
from modules.solov2.solov2 import SOLOv2
import warnings
from etaprogress.progress import ProgressBar
from adet.config import get_cfg
warnings.filterwarnings("ignore")

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg

def main(solo):
    # parse options
    opt = TrainOptions().parse()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load the dataset
    dataloader, _ = get_loader(device=device, \
                                    root=opt.coco+'train2017', \
                                        json=opt.coco+'annotations/instances_train2017.json', \
                                            batch_size=opt.batch_size, \
                                                shuffle=False, \
                                                    num_workers=0)

    # create trainer for our model
    trainer = OlieTrainer(opt, solo)

    # create tool for counting iterations
    iter_counter = IterationCounter(opt, len(dataloader))

    # create tool for visualization
    visualizer = Visualizer(opt)

    for epoch in iter_counter.training_epochs():
        iter_counter.record_epoch_start(epoch)
        total = len(dataloader)
        bar = ProgressBar(total, max_width=80)
        for i, data_i in enumerate(dataloader, start=iter_counter.epoch_iter):
            bar.numerator = i+1
            print(bar, end='\r')

            iter_counter.record_one_iteration()

            # Training
            # train generator
            if i % opt.D_steps_per_G == 0:
                trainer.run_generator_one_step(data_i)

            # train discriminator
            trainer.run_discriminator_one_step(data_i)

            # Visualizations
            if iter_counter.needs_printing():
                losses = trainer.get_latest_losses()
                visualizer.print_current_errors(epoch, iter_counter.epoch_iter,
                                                losses, iter_counter.time_per_iter)
                visualizer.plot_current_errors(losses, iter_counter.total_steps_so_far)

            if iter_counter.needs_displaying():
                
                visuals = OrderedDict([('input_label', trainer.get_semantics().max(dim=1)[1].cpu().unsqueeze(1)),
                                    ('synthesized_image', trainer.get_latest_generated()),
                                    ('real_image', data_i['image']),
                                    ('masked', trainer.get_mask())])
    
                if not opt.no_instance:
                        visuals['instance'] = trainer.get_semantics()[:,35].cpu()


                visualizer.display_current_results(visuals, epoch, iter_counter.total_steps_so_far)

            if iter_counter.needs_saving():
                print('saving the latest model (epoch %d, total_steps %d)' %
                    (epoch, iter_counter.total_steps_so_far))
                trainer.save('latest')
                iter_counter.record_current_iter()

        trainer.update_learning_rate(epoch)
        iter_counter.record_epoch_end()

        if epoch % opt.save_epoch_freq == 0 or \
        epoch == iter_counter.total_epochs:
            print('saving the model at the end of epoch %d, iters %d' %
                (epoch, iter_counter.total_steps_so_far))
            trainer.save('latest')
            trainer.save(epoch)

    print('Training was successfully finished.')

if __name__ == '__main__':
    args = TrainOptions().parse()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = setup_cfg(args)

    solo = SOLOv2(cfg=cfg).to(device)
    checkpointer = DetectionCheckpointer(solo)
    checkpointer.load(cfg.MODEL.WEIGHTS)

    main(solo=solo.eval())