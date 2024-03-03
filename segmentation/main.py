"""
Author: Benny
Date: Nov 2019
"""
import argparse
import os
import torch
import datetime
import logging
import sys
import importlib
import shutil
import provider
import numpy as np
import torch.optim as optim
from timm.scheduler import CosineLRScheduler
from pathlib import Path
from tqdm import tqdm
from dataset import PartNormalDataset
import torch.nn.functional as F
from front3d_semantic_dataset import Front3DSemanticDataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

# seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
#                'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37],
#                'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49],
#                'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}


# seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
# for cat in seg_classes.keys():
#     for label in seg_classes[cat]:
#         seg_label_to_cat[label] = cat

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def masked_cross_entropy(criterion, targets, logits_pred, mask, num_classes=19):
    # Apply the mask to the targets and logits
    targets_masked = targets * mask
    targets_masked = targets_masked.squeeze(-1)
    logits_pred_masked = logits_pred * mask

    logits_pred_masked = logits_pred_masked.reshape(-1, num_classes)
    targets_masked = targets_masked.reshape(-1)

    # Compute cross-entropy loss with the masked targets and logits
    loss = criterion(logits_pred_masked, targets_masked.long())

    return loss


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y


def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pt', help='model name')
    parser.add_argument('--batch_size', type=int, default=8, help='batch Size during training')
    parser.add_argument('--epoch', default=300, type=int, help='epoch to run')
    parser.add_argument('--warmup_epoch', default=10, type=int, help='warmup epoch')
    parser.add_argument('--learning_rate', default=0.0002, type=float, help='initial learning rate')
    parser.add_argument('--gpu', type=str, default='0', help='specify GPU devices')
    # parser.add_argument('--optimizer', type=str, default='AdamW', help='Adam or SGD')
    parser.add_argument('--log_dir', type=str, default='./exp', help='log path')
    # parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--npoint', type=int, default=2048, help='point Number')
    parser.add_argument('--normal', action='store_true', default=False, help='use normals')
    # parser.add_argument('--step_size', type=int, default=20, help='decay step for lr decay')
    # parser.add_argument('--lr_decay', type=float, default=0.5, help='decay rate for lr decay')
    parser.add_argument('--ckpts', type=str, default=None, help='ckpts')
    parser.add_argument('--root', type=str, default='../data/shapenetcore_partanno_segmentation_benchmark_v0_normal/', help='data root')
    return parser.parse_args()

def intersectionAndUnionGPU(output, target, K):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    # assert output.dim() in [1, 2, 3]
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)

    # Ignore class 0 in both output and target
    ignore_mask = target != 0
    output = output[ignore_mask]
    target = target[ignore_mask]

    # Calculate the intersection and union areas for each class
    intersection = output[output == target]
    area_intersection = torch.histc(
        intersection, bins=K - 1, min=1, max=K - 1
    )  # Intersection (ignore class 0)
    area_output = torch.histc(
        output, bins=K - 1, min=1, max=K - 1
    )  # Area of predicted regions (ignore class 0)
    area_target = torch.histc(
        target, bins=K - 1, min=1, max=K - 1
    )  # Area of target regions (ignore class 0)

    # Calculate the union area by adding the areas of output and target and subtracting the intersection
    area_union = area_output + area_target - area_intersection

    return area_intersection, area_union, area_target


def output_metrics(x, pred, num_classes=19):

    target = x

    probabilities = F.softmax(pred, dim=1)
    predicted_labels = torch.argmax(probabilities, dim=1)

    intersection, union, target = intersectionAndUnionGPU(
        predicted_labels, target, num_classes
    )
    metrics = {}
    metrics["intersection"] = intersection
    metrics["union"] = union
    metrics["target"] = target
    # metrics["iou_val"] = iou_val

    return metrics
    

def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('part_seg')
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    root = args.root

    # TRAIN_DATASET = PartNormalDataset(root=root, npoints=args.npoint, split='trainval', normal_channel=args.normal)

    dataset_split = '/wild6d_data/zubair/nerf_rpn/front3d_rpn_data/front3d_split.npz'
    with np.load(dataset_split) as split:
        train_scenes = split["train_scenes"]
        test_scenes = split["test_scenes"]
        val_scenes = split["val_scenes"]

    features_path = '/wild6d_data/zubair/nerf_rpn/front3d_rpn_data/features'
    sem_feat_path = '/wild6d_data/zubair/nerf_rpn/front3d_rpn_data/voxel_front3d'
    TRAIN_DATASET = Front3DSemanticDataset(
        features_path=features_path,
        sem_feat_path=sem_feat_path,
        scene_list=train_scenes,
        preload=False,
        percent_train=1.0,
    )

    TEST_DATASET = Front3DSemanticDataset(
        features_path=features_path,
        sem_feat_path=sem_feat_path,
        scene_list=test_scenes,
        preload=False,
        percent_train=1.0,
    )



    # trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=True)

    trainDataLoader =  torch.utils.data.DataLoader(
        TRAIN_DATASET,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=TRAIN_DATASET.collate_fn,
    )

    # TEST_DATASET = PartNormalDataset(root=root, npoints=args.npoint, split='test', normal_channel=args.normal)

    # testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=10)

    testDataLoader =  torch.utils.data.DataLoader(
        TEST_DATASET,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=TEST_DATASET.collate_fn,
    )

    log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    log_string("The number of test data is: %d" % len(TEST_DATASET))

    # num_classes = 16
    num_part = 20

    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)
    shutil.copy('models/%s.py' % args.model, str(exp_dir))
    # shutil.copy('models/pointnet2_utils.py', str(exp_dir))

    classifier = MODEL.get_model(num_part).cuda()
    criterion = MODEL.get_loss().cuda()
    classifier.apply(inplace_relu)
    print('# generator parameters:', sum(param.numel() for param in classifier.parameters()))
    start_epoch = 0

    if args.ckpts is not None:
        classifier.load_model_from_ckpt(args.ckpts)

## we use adamw and cosine scheduler
    def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
        decay = []
        no_decay = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights
            if len(param.shape) == 1 or name.endswith(".bias") or 'token' in name or name in skip_list:
                        # print(name)
                no_decay.append(param)
            else:
                decay.append(param)
        return [
                    {'params': no_decay, 'weight_decay': 0.},
                    {'params': decay, 'weight_decay': weight_decay}]

    param_groups = add_weight_decay(classifier, weight_decay=0.05)
    optimizer = optim.AdamW(param_groups, lr= args.learning_rate, weight_decay=0.05 )

    scheduler = CosineLRScheduler(optimizer,
                                  t_initial=args.epoch,
                                  t_mul=1,
                                  lr_min=1e-6,
                                  decay_rate=0.1,
                                  warmup_lr_init=1e-6,
                                  warmup_t=args.warmup_epoch,
                                  cycle_limit=1,
                                  t_in_epochs=True)

    best_acc = 0
    global_epoch = 0
    best_class_avg_iou = 0
    best_inctance_avg_iou = 0

    classifier.zero_grad()
    for epoch in range(start_epoch, args.epoch):
        mean_correct = []

        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        '''Adjust learning rate and BN momentum'''

        classifier = classifier.train()
        loss_batch = []
        num_iter = 0
        '''learning one epoch'''
        # for i, (points, label, target) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):

        # for i, (points, label, target) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):

        for i, data in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):

            grid, alpha, out_sem = data

            grid = grid[0]
            alpha = alpha[0]
            out_sem = out_sem[0]

            mask = alpha > 0.01
            points = grid[mask, :]
            out_sem = out_sem[mask]

            print("points", points.shape, out_sem.shape)

            # num_iter += 1
            # points = points.data.numpy()
            # points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            # points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = points.unsqueeze(0)

            out_sem = out_sem.unsqueeze(0)


            points, target = points.float().cuda(), out_sem.long().cuda()
            points = points.transpose(2, 1)

            print("points", points.shape, target.shape)

            # seg_pred = classifier(points, to_categorical(label, num_classes))

            seg_pred = classifier(points)
            seg_pred = seg_pred.contiguous().view(-1, num_part)
            target = target.view(-1, 1)[:, 0]
            pred_choice = seg_pred.data.max(1)[1]

            correct = pred_choice.eq(target.data).cpu().sum()
            mean_correct.append(correct.item() / (args.batch_size * args.npoint))

            mask = target > 0
            seg_pred = seg_pred[mask, :]
            target = target[mask]
            loss = criterion(seg_pred, target)
            loss.backward()
            optimizer.step()
            loss_batch.append(loss.detach().cpu())

            if num_iter == 1:

                torch.nn.utils.clip_grad_norm_(classifier.parameters(), 10, norm_type=2)
                num_iter = 0
                optimizer.step()
                classifier.zero_grad()

        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)

        train_instance_acc = np.mean(mean_correct)
        loss1 = np.mean(loss_batch)
        log_string('Train accuracy is: %.5f' % train_instance_acc)
        log_string('Train loss: %.5f' % loss1)
        log_string('lr: %.6f' % optimizer.param_groups[0]['lr'])

        with torch.no_grad():
            # test_metrics = {}
            # total_correct = 0
            # total_seen = 0
            # total_seen_class = [0 for _ in range(num_part)]
            # total_correct_class = [0 for _ in range(num_part)]
            # shape_ious = {cat: [] for cat in seg_classes.keys()}
            # seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}

            # for cat in seg_classes.keys():
            #     for label in seg_classes[cat]:
            #         seg_label_to_cat[label] = cat

            classifier = classifier.eval()

            # for batch_id, (points, label, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):

            intersection_meter = AverageMeter()
            union_meter = AverageMeter()
            target_meter = AverageMeter()


            for batch_id, data in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):

                grid, alpha, out_sem = data

                grid = grid[0]
                alpha = alpha[0]
                out_sem = out_sem[0]

                mask = alpha > 0.01
                points = grid[mask, :]
                out_sem = out_sem[mask]
                

                # cur_batch_size, NUM_POINT = points.size()

                points = points.unsqueeze(0)

                out_sem = out_sem.unsqueeze(0)


                points, target = points.float().cuda(), out_sem.long().cuda()
                points = points.transpose(2, 1)

                outputs = output_metrics(target, classifier(points), num_classes=num_part)
                

                intersection = outputs["intersection"]
                union = outputs["union"]
                target = outputs["target"]
                # iou_val = outputs["iou_val"]
                intersection, union, target = (
                    intersection.cpu().numpy(),
                    union.cpu().numpy(),
                    target.cpu().numpy(),
                )
                intersection_meter.update(intersection), union_meter.update(
                    union
                ), target_meter.update(target)

                accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)

        iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
        accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
        allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

        # print("iou_class", iou_class, "accuracy_class", accuracy_class)
        mIoU = np.mean(iou_class)
        mAcc = np.mean(accuracy_class)

        log_string('Epoch %d test Accuracy: %f  Class avg mIOU: %f   Inctance avg mIOU: %f' % (
            epoch + 1, allAcc, mIoU, mAcc))


        state = {
            'epoch': epoch,
            'train_acc': train_instance_acc,
            'model_state_dict': classifier.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        savepath = str(checkpoints_dir) + '/best_model.pth'
        torch.save(state, savepath)
        log_string('Saving model....')
        global_epoch += 1


if __name__ == '__main__':
    args = parse_args()
    main(args)