# encoding: utf-8
"""
@author:  merlin
@contact: merlinarer@gmail.com
"""

import argparse
import os
import torch
from torch import nn
from torch.autograd import Variable
import time
from torchvision import transforms
from torch.backends import cudnn
from torch.utils.data import DataLoader
import logging

from config import cfg
from dataloader import MyDataset
from modeling import build_model

from utils.logger import setup_logger
from utils.checkpoint import save_checkpoint
from solver.optimizer import make_optimizer
from solver.lr_scheduler import WarmupMultiStepLR

from modeling.losses.build import get_loss

def train(cfg):
    # logger
    logger = logging.getLogger(name="merlin.baseline.train")
    logger.info("training...")

    # transform
    transform_train_list = [
        # transforms.RandomResizedCrop(size=128, scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
        transforms.Resize(size=cfg.INPUT.SIZE_TRAIN, interpolation=1),
        transforms.Pad(32),
        transforms.RandomCrop(cfg.INPUT.SIZE_TRAIN),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

    transform_val_list = [
        transforms.Resize(size=cfg.INPUT.SIZE_TEST, interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

    # prepare dataset
    train_dataset = MyDataset(root=cfg.DATA.ROOT, transform=transforms.Compose(transform_train_list), type='train')
    val_dataset = MyDataset(root=cfg.DATA.ROOT, transform=transforms.Compose(transform_val_list), type='val')
    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.SOLVER.BATCH_SIZE,
                              shuffle=True,
                              num_workers=8,
                              pin_memory=False)
    val_loader = DataLoader(val_dataset,
                              batch_size=cfg.SOLVER.BATCH_SIZE,
                              shuffle=True,
                              num_workers=8,
                              pin_memory=False)
    num_classes = cfg.MODEL.HEADS.NUM_CLASSES

    # prepare model
    model = build_model(cfg, num_classes)
    model = model.cuda()
    model = nn.DataParallel(model)

    # prepare solver
    optimizer = make_optimizer(cfg, model)
    scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
                                  cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)

    start_epoch = 0

    # Train and val
    since = time.time()
    for epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCHS):
        model.train(True)
        logger.info("Epoch {}/{}".format(epoch, cfg.SOLVER.MAX_EPOCHS - 1))
        logger.info('-' * 10)

        running_loss = 0.0
        # Iterate over data
        it = 0
        running_acc = 0
        for data in train_loader:
            it += 1
            # get the inputs
            inputs, labels = data
            now_batch_size, c, h, w = inputs.shape
            if now_batch_size < cfg.SOLVER.BATCH_SIZE:  # skip the last batch
                continue

            # wrap them in Variable
            inputs = Variable(inputs.cuda().detach())
            labels = Variable(labels.cuda().detach())

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            out = model(inputs)
            loss_dict = get_loss(cfg, outs=out, label=labels)
            loss = sum(loss_dict.values())

            loss.backward()
            optimizer.step()
            scheduler.step()

            # statistics
            with torch.no_grad():
                _, preds = torch.max(out['pred_class_logits'], 1)
                running_loss += loss
                running_acc += torch.sum(preds == labels.data).float().item() / cfg.SOLVER.BATCH_SIZE

            if it % 50 == 0:
                logger.info(
                    'epoch {}, iter {}, loss: {:.3f}, acc: {:.3f}, lr: {:.5f}'.format(
                        epoch, it, running_loss / it, running_acc / it,
                        optimizer.param_groups[0]['lr']))

        epoch_loss = running_loss / it
        epoch_acc = running_acc / it

        logger.info('epoch {} loss: {:.4f} Acc: {:.4f}'.format(epoch, epoch_loss, epoch_acc))

        # save checkpoint
        if epoch % cfg.SOLVER.CHECKPOINT_PERIOD == 0:
            checkpoint = {'epoch': epoch + 1,
                          'model': model.module.state_dict() if (len(cfg.MODEL.DEVICE_ID) - 2) > 1 else model.state_dict(),
                          'optimizer': optimizer.state_dict()
                          }
            save_checkpoint(checkpoint, epoch, cfg)

        # evaluate
        if epoch % cfg.SOLVER.EVAL_PERIOD == 0:
            logger.info('evaluate...')
            model.train(False)

            total = 0.0
            correct = 0.0
            for data in val_loader:
                inputs, labels = data
                inputs = Variable(inputs.cuda().detach())
                labels = Variable(labels.cuda().detach())
                with torch.no_grad():
                    out = model(inputs)
                    _, preds = torch.max(out['pred_class_logits'], 1)
                    c = (preds == labels).squeeze()
                    total += c.size(0)
                    correct += c.float().sum().item()
            acc = correct / total
            logger.info('eval acc:{:.4f}'.format(acc))

        time_elapsed = time.time() - since
        logger.info('Training complete in {:.0f}m {:.0f}s\n'.format(
            time_elapsed // 60, time_elapsed % 60))

    return model


def main():
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger(name="merlin.baseline", output=output_dir)
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DEVICE == "cuda":
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    cudnn.benchmark = True
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    # logger.info("Using {} GPUS".format(num_gpus))
    train(cfg)


if __name__ == '__main__':
    main()
