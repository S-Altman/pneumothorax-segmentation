import torch
from torch.nn.utils import clip_grad_norm_
torch.multiprocessing.set_sharing_strategy('file_system')

from utils.mask_functions import compute_sdm_and_wmap
# from utils.metrics import ConfusionMatrix, dice, jaccard, hausdorff_distance_95, avg_surface_distance


import pandas as pd
import numpy as np

from tqdm import tqdm
from pathlib import Path
from medpy.metric.binary import dc, jc, hd, asd

import heapq
from collections import defaultdict

import warnings
warnings.filterwarnings('ignore')

class Learning():
    def __init__(self,
                 optimizer, 
                 binarizer_fn,
                 loss_fn,
                 eval_fn,
                 device,
                 n_epoches,
                 scheduler,    
                 freeze_model,
                 grad_clip,
                 grad_accum,
                 early_stopping,
                 validation_frequency,
                 calculation_name,
                 best_checkpoint_folder,
                 checkpoints_history_folder,
                 checkpoints_topk,
                 logger
        ):
        self.logger = logger

        self.optimizer = optimizer
        self.binarizer_fn = binarizer_fn
        self.loss_fn = loss_fn
        self.eval_fn = eval_fn

        self.device = device
        self.n_epoches = n_epoches
        self.scheduler = scheduler
        self.freeze_model = freeze_model
        self.grad_clip = grad_clip
        self.grad_accum = grad_accum
        self.early_stopping = early_stopping
        self.validation_frequency = validation_frequency

        self.calculation_name = calculation_name
        self.best_checkpoint_path = Path(
            best_checkpoint_folder, 
            '{}.pth'.format(self.calculation_name)
        )
        self.checkpoints_history_folder = Path(checkpoints_history_folder)
        self.checkpoints_topk = checkpoints_topk
        self.score_heap = []
        self.summary_file = Path(self.checkpoints_history_folder, 'summary.csv')     
        if self.summary_file.is_file():
            self.best_score = pd.read_csv(self.summary_file).best_metric.max()
            logger.info('Pretrained best score is {:.5}'.format(self.best_score))
        else:
            self.best_score = 0.0
        self.best_epoch = -1

    def train_epoch(self, model, loader):
        tqdm_loader = tqdm(loader)
        current_loss_mean = 0

        for batch_idx, (imgs, labels) in enumerate(tqdm_loader):
            loss, predicted = self.batch_train(model, imgs, labels, batch_idx)

            # just slide average
            current_loss_mean = (current_loss_mean * batch_idx + loss) / (batch_idx + 1)

            tqdm_loader.set_description('loss: {:.4} lr:{:.6}'.format(
                current_loss_mean, self.optimizer.param_groups[0]['lr']))
        return current_loss_mean

    def batch_train(self, model, batch_imgs, batch_labels, batch_idx):
        # batch_imgs, batch_labels = batch_imgs.to(self.device), batch_labels.to(self.device)
        batch_imgs = batch_imgs.to(self.device)

        predicted = model(batch_imgs)
        loss = self.loss_fn(predicted, batch_labels)

        loss.backward()
        if batch_idx % self.grad_accum == self.grad_accum - 1:
            clip_grad_norm_(model.parameters(), self.grad_clip)
            self.optimizer.step()
            self.optimizer.zero_grad()
        return loss.item(), predicted

    def valid_epoch(self, model, loader):
        tqdm_loader = tqdm(loader)
        current_score_mean = 0
        used_thresholds = self.binarizer_fn.thresholds
        # metrics = defaultdict(float)
        dices = defaultdict(float)
        jacs = defaultdict(float)
        hd95s = defaultdict(float)
        asds = defaultdict(float)

        sample_count = 0

        for batch_idx, (imgs, labels) in enumerate(tqdm_loader):
            with torch.no_grad():
                sample_count += imgs.size(0)
                predicted_probas = self.batch_valid(model, imgs)
                labels = labels.to(self.device)
                # labels = labels.numpy()
                mask_generator = self.binarizer_fn.transform(predicted_probas)
                for current_thr, current_mask in zip(used_thresholds, mask_generator):
                    current_metric = self.eval_fn(current_mask, labels).item()
                    # cm = current_mask.detach().cpu().numpy()
                    # current_mask = current_mask.detach().cpu().numpy()
                    # confusion_matrix = ConfusionMatrix(test=current_mask, reference=labels)

                    current_thr = tuple(current_thr)

                    # dice_batch = dice(confusion_matrix=confusion_matrix, nan_for_nonexisting=False)
                    # dices[current_thr] += dice_batch
                    dices[current_thr] = (dices[current_thr] * batch_idx + current_metric) / (batch_idx + 1)

                    # jac_batch = jaccard(confusion_matrix=confusion_matrix, nan_for_nonexisting=False)
                    jac_batch = current_metric / (2 - current_metric)
                    jacs[current_thr] += jac_batch
                    # jacs[current_thr] = (jacs[current_thr] * batch_idx + jac_batch) / (batch_idx + 1)

                    # hd95_batch = hausdorff_distance_95(confusion_matrix=confusion_matrix, nan_for_nonexisting=False)
                    # hd95s[current_thr] += hd95_batch
                    # # hd95s[current_thr] = (hd95s[current_thr] * batch_idx + hd95_batch) / (batch_idx + 1)
                    #
                    # asd_batch = avg_surface_distance(confusion_matrix=confusion_matrix, nan_for_nonexisting=False)
                    # asds[current_thr] += asd_batch
                    # # asds[current_thr] = (asds[current_thr] * batch_idx + asd_batch) / (batch_idx + 1)



                best_threshold = max(dices, key=dices.get)
                best_metric = dices[best_threshold]
                tqdm_loader.set_description('score: {:.5} on {}'.format(best_metric, best_threshold))

        best_jac = jacs[best_threshold] / sample_count
        # best_hd95 = hd95s[best_threshold] / sample_count
        # best_asd = asds[best_threshold] / sample_count

        return dices, best_metric, best_jac

    def batch_valid(self, model, batch_imgs):
        batch_imgs = batch_imgs.to(self.device)
        predicted = model(batch_imgs)
        if isinstance(predicted, list):
            predicted[0] = torch.sigmoid(predicted[0])
            predicted[1] = torch.tanh(predicted[1])
        else:
            predicted = torch.sigmoid(predicted)
        return predicted

    def process_summary(self, metrics, epoch, best_jac):
        best_threshold = max(metrics, key=metrics.get)
        best_metric = metrics[best_threshold]

        epoch_summary = pd.DataFrame.from_dict([metrics])
        epoch_summary['epoch'] = epoch
        epoch_summary['best_metric'] = best_metric
        epoch_summary['jac'] = best_jac
        epoch_summary = epoch_summary[['epoch', 'best_metric', 'jac'] + list(metrics.keys())]
        epoch_summary.columns = [str(col) for col in epoch_summary.columns]
        
        self.logger.info('{} epoch: \t Score: {:.5}\t Params: {}'.format(epoch, best_metric, best_threshold))

        if not self.summary_file.is_file():
            epoch_summary.to_csv(self.summary_file, index=False)
        else:
            summary = pd.read_csv(self.summary_file)
            summary = summary.append(epoch_summary).reset_index(drop=True)
            summary.to_csv(self.summary_file, index=False)  

    @staticmethod
    def get_state_dict(model):
        if type(model) == torch.nn.DataParallel:
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()
        return state_dict

    def post_processing(self, score, epoch, model):
        if self.freeze_model:
            return

        checkpoints_history_path = Path(
            self.checkpoints_history_folder, 
            '{}_epoch{}.pth'.format(self.calculation_name, epoch)
        )

        torch.save(self.get_state_dict(model), checkpoints_history_path)
        heapq.heappush(self.score_heap, (score, checkpoints_history_path))
        if len(self.score_heap) > self.checkpoints_topk:
            _, removing_checkpoint_path = heapq.heappop(self.score_heap)
            removing_checkpoint_path.unlink()
            self.logger.info('Removed checkpoint is {}'.format(removing_checkpoint_path))
        if score > self.best_score:
            self.best_score = score
            self.best_epoch = epoch
            torch.save(self.get_state_dict(model), self.best_checkpoint_path)
            self.logger.info('best model: {} epoch - {:.5}'.format(epoch, score))

        if self.scheduler.__class__.__name__ == 'ReduceLROnPlateau':
            self.scheduler.step(score)
        else:
            self.scheduler.step()

    def run_train(self, model, train_dataloader, valid_dataloader):
        model.to(self.device)
        for epoch in range(self.n_epoches):
            if not self.freeze_model:
                self.logger.info('{} epoch: \t start training....'.format(epoch))
                model.train()
                train_loss_mean = self.train_epoch(model, train_dataloader)
                self.logger.info('{} epoch: \t Calculated train loss: {:.5}'.format(epoch, train_loss_mean))

            if epoch % self.validation_frequency != (self.validation_frequency - 1):
                self.logger.info('skip validation....')
                continue

            self.logger.info('{} epoch: \t start validation....'.format(epoch))
            model.eval()
            metrics, score, best_jac = self.valid_epoch(model, valid_dataloader)

            self.process_summary(metrics, epoch, best_jac)

            self.post_processing(score, epoch, model)

            if epoch - self.best_epoch > self.early_stopping:
                self.logger.info('EARLY STOPPING')
                break

        self.logger.info('best model: {} epoch - {:.5}'.format(self.best_epoch, self.best_score))

        return self.best_score, self.best_epoch
        
