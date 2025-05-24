from os import listdir, makedirs, path
from typing import Callable, Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import piq
import torch
import wandb
from PIL import Image
from einops import rearrange
from lightning import LightningModule
from torch import Tensor
from torch.optim import AdamW, Optimizer
from accelerate import PartialState

OptimizerCallable = Callable[[Iterable], Optimizer]

from genie.modules import LatentActionModel, ControlAwareLatentActionModel, LatentActionModel_MultiView, DINOLatentActionModel, UncontrolledDINOLatentActionModel, ControllableDINOLatentActionModel

import logging
logging.basicConfig(format='%(message)s', level=logging.INFO)


class Genie(LightningModule):
    """
    Generative Interactive Environment model from Bruce et al. (2024).
    The model is composed of:
    - A (pre-trained) tokenizer.
    - A latent action model that build a (quantized) dictionary of latent actions.
    - A dynamics Model that predicts the next frame given the current frame and the latent action.
    """

    def __init__(
            self,
            image_channels: int = 3,
            # Latent action model
            lam_model_dim: int = 512,
            lam_latent_dim: int = 32,
            lam_num_latents: int = 8,
            lam_patch_size: int = 16,
            lam_enc_blocks: int = 8,
            lam_dec_blocks: int = 8,
            lam_num_heads: int = 8,
            lam_dropout: float = 0.0,
            vq_beta: float = 0.25,
            log_interval: int = 1000,
            log_path: str = "log_imgs",
            task_name: str = 'lam_openx',
            optimizer: OptimizerCallable = AdamW,
            make_data_pair: bool = False,
            control_aware: bool = False,
            stage_2: bool = False,
            multi_view: bool = False,
    ) -> None:
        super(Genie, self).__init__()
        if control_aware:
            self.lam = ControlAwareLatentActionModel(
                in_dim=image_channels,
                model_dim=lam_model_dim,
                latent_dim=lam_latent_dim,
                num_latents=lam_num_latents,
                patch_size=lam_patch_size,
                enc_blocks=lam_enc_blocks,
                dec_blocks=lam_dec_blocks,
                num_heads=lam_num_heads,
                dropout=lam_dropout,
                stage_2=stage_2,
            )
            # if stage_2:
            #     lam_ckpt = torch.load('/cpfs01/user/buqingwen/OmniEmbodiment/logs/stage1_ckpts_openx_v2/epoch=1-step=215000.ckpt')['state_dict']
            #     stage1_ckpt = {}
            #     for key in lam_ckpt.keys():
            #         if 'action_latent' in key or 'vq' in key:
            #             stage1_ckpt[key.replace("lam.", "")] = lam_ckpt[key]
            #     self.lam.load_state_dict(stage1_ckpt, strict=False)
        elif multi_view:
            self.lam = LatentActionModel_MultiView(
                in_dim=image_channels,
                model_dim=lam_model_dim,
                latent_dim=lam_latent_dim,
                num_latents=lam_num_latents,
                patch_size=lam_patch_size,
                enc_blocks=lam_enc_blocks,
                dec_blocks=lam_dec_blocks,
                num_heads=lam_num_heads,
                dropout=lam_dropout
            )
        else:
            self.lam = LatentActionModel(
                in_dim=image_channels,
                model_dim=lam_model_dim,
                latent_dim=lam_latent_dim,
                num_latents=lam_num_latents,
                patch_size=lam_patch_size,
                enc_blocks=lam_enc_blocks,
                dec_blocks=lam_dec_blocks,
                num_heads=lam_num_heads,
                dropout=lam_dropout
            )
        self.lam_num_latents = lam_num_latents
        self.vq_beta = vq_beta
        self.log_interval = log_interval
        self.log_path = log_path
        self.optimizer = optimizer
        self.make_data_pair = make_data_pair
        self.control_aware = control_aware
        self.stage_2 = stage_2

        self.save_hyperparameters()

        self.task_name = task_name
        self.distributed_state = PartialState()
        if self.distributed_state.is_main_process:
            wandb.init(name=task_name, reinit=True)

    def shared_step(self, batch: Dict) -> Tuple:
        # batch: keys['videos', 'task_instruction', 'action', 'dataset_names']

        outputs = self.lam(batch)
        gt_future_frames = batch["videos"][:, 1:]

        # Compute loss
        mse_loss = ((gt_future_frames - outputs["recon"]) ** 2).mean()
        if "recon_view2" in outputs.keys():
            mse_loss += ((batch['videos_view2'][:, 1:] - outputs["recon_view2"]) ** 2).mean()

        q_loss = ((outputs["emb"].detach() - outputs["z"]) ** 2).mean()
        # if self.stage_2:
        #     commit_loss = torch.tensor(0).to(self.lam.device)
        # else:
        commit_loss = ((outputs["emb"] - outputs["z"].detach()) ** 2).mean()

        loss = mse_loss + q_loss + self.vq_beta * commit_loss
        
        if self.stage_2:
            q_loss_action = ((outputs["emb_action"].detach() - outputs["z_action"]) ** 2).mean()
            commit_loss_action = ((outputs["emb_action"]- outputs["z_action"].detach()) ** 2).mean()

            loss = loss + q_loss_action + self.vq_beta * commit_loss_action


        # Compute monitoring measurements
        gt = gt_future_frames.clamp(0, 1).reshape(-1, *gt_future_frames.shape[2:])
        recon = outputs["recon"].clamp(0, 1).reshape(-1, *outputs["recon"].shape[2:])
        psnr = piq.psnr(gt, recon).mean()
        ssim = piq.ssim(gt, recon).mean()

        # Compute code usage
        unique, counts = torch.unique(outputs["indices"], return_counts=True)
        index_counts = torch.zeros(self.lam_num_latents, dtype=torch.long).cuda()
        index_counts[unique] = counts
        code_usage = (index_counts != 0).float().mean()

        loss_logs = (
            ("mse_loss", mse_loss),
            ("q_loss", q_loss),
            ("commit_loss", commit_loss),
            ("psnr", psnr),
            ("ssim", ssim),
            ("code_usage", code_usage),
        )
        if self.stage_2:
            loss_logs = loss_logs + (("q_loss_action", q_loss_action),)
            loss_logs = loss_logs + (("commit_loss_action", commit_loss_action),)
            
        return outputs, loss, loss_logs

    def training_step(self, batch: Dict, batch_idx: int) -> Tensor:
        # Compute the training loss
        outputs, loss, aux_losses = self.shared_step(batch)


        # Log the training loss
        self.log_dict(
            {**{"train_loss": loss}, **{f"train/{k}": v for k, v in aux_losses}},
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True
        )

        if self.distributed_state.is_main_process:
            wandb.log({**{"train_loss": loss}, **{f"train/{k}": v for k, v in aux_losses}})

        if batch_idx % self.log_interval == 0:  # Start of the epoch
            self.log_images(batch, outputs, "train")

        return loss

    # @torch.no_grad()
    # def validation_step(self, batch: Dict, batch_idx: int) -> Tensor:
    #     # Compute the validation loss
    #     outputs, loss, aux_losses = self.shared_step(batch)
    #
    #     # Log the validation loss
    #     self.log_dict(
    #         {**{"val_loss": loss}, **{f"val/{k}": v for k, v in aux_losses}},
    #         prog_bar=True,
    #         logger=True,
    #         on_step=True,
    #         on_epoch=True,
    #         sync_dist=True
    #     )
    #
    #     if batch_idx % self.log_interval == 0:  # Start of the epoch
    #         self.log_images(batch, outputs, "val")
    #     return loss

    @torch.no_grad()
    def test_step(self, batch: Dict, batch_idx: int) -> Tensor:
        # Compute the test loss
        outputs, loss, aux_losses = self.shared_step(batch)

        # Log the test loss
        self.log_dict(
            {**{"test_loss": loss}, **{f"test/{k}": v for k, v in aux_losses}},
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True
        )

        self.log_images(batch, outputs, "test")

        return loss

    def log_images(self, batch: Dict, outputs: Dict, split: str) -> None:
        # gt_seq = batch["videos"][0][1:].clamp(0, 1).cpu()
        gt_seq = batch["videos"][0].clamp(0, 1).cpu()
        recon_seq = outputs["recon"][0].clamp(0, 1).cpu()
        compare_seq = torch.cat([gt_seq[:1], gt_seq[1:], recon_seq], dim=3)
        if "recon_view2" in outputs.keys():
            gt_seq = batch["videos_view2"][0].clamp(0, 1).cpu()
            recon_seq = outputs["recon_view2"][0].clamp(0, 1).cpu()
            compare_seq_view2 = torch.cat([gt_seq[:1], gt_seq[1:], recon_seq], dim=3)
            compare_seq = torch.cat([compare_seq, compare_seq_view2], dim=2)
        compare_seq = rearrange(compare_seq * 255, "t c h w -> h (t w) c")
        compare_seq = compare_seq.detach().numpy().astype(np.uint8)
        if split == 'test':
            import random
            idx = random.randint(0,100)
            img_path = path.join(self.log_path, f"vis_{self.task_name}", f"{split}_step{idx:06}.png")
        else:
            img_path = path.join(self.log_path, f"vis_{self.task_name}", f"{split}_step{self.global_step:06}.png")
        makedirs(path.dirname(img_path), exist_ok=True)
        img = Image.fromarray(compare_seq)
        print(img_path)
        img.save(img_path)

    def on_train_epoch_end(self):
        self.lam.vq.random_restart()
        self.lam.vq.reset_usage()

    def on_test_epoch_end(self):
        if self.make_data_pair:
            completed = len(listdir("output_pairs"))
            todo_name = listdir("../data/retro")[completed]
            makedirs(f"output_pairs/{todo_name}")
            top_indices = torch.topk(self.lam.vq.usage, 16, largest=True, sorted=True).indices
            top_latents = self.lam.vq.codebook(top_indices)
            torch.save(top_latents, f"output_pairs/{todo_name}/top_16.pt")
            with open(f"output_pairs/{todo_name}/top_16.txt", "w") as f:
                f.write(" ".join([str(i) for i in top_indices.tolist()]))

        self.plot_usage_distribution(self.lam.vq.usage, "unsorted_usage")
        self.plot_usage_distribution(self.lam.vq.usage.sort().values, "sorted_usage")

    def plot_usage_distribution(self, usage, filename):
        data = usage.cpu().numpy()
        n = 1
        for n in range(1, 10):
            if (2 ** n) ** 2 <= len(data) < (2 ** (n + 1)) ** 2:
                break
        data = data.reshape(2 ** n, -1)
        fig, ax = plt.subplots()
        cax = ax.matshow(data, interpolation="nearest")
        fig.colorbar(cax)
        plt.axis("off")
        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig(f"{filename}.png", bbox_inches="tight", pad_inches=0.0)
        plt.close()

    def configure_optimizers(self) -> Optimizer:
        optim = self.optimizer(self.parameters())
        return optim



class DINO_LAM(LightningModule):
    """
    Generative Interactive Environment model from Bruce et al. (2024).
    The model is composed of:
    - A (pre-trained) tokenizer.
    - A latent action model that build a (quantized) dictionary of latent actions.
    - A dynamics Model that predicts the next frame given the current frame and the latent action.
    """

    def __init__(
            self,
            image_channels: int = 3,
            # Latent action model
            lam_model_dim: int = 512,
            lam_latent_dim: int = 32,
            lam_num_latents: int = 8,
            lam_patch_size: int = 16,
            lam_enc_blocks: int = 8,
            lam_dec_blocks: int = 8,
            lam_num_heads: int = 8,
            lam_dropout: float = 0.0,
            vq_beta: float = 0.25,
            log_interval: int = 1000,
            log_path: str = "log_imgs",
            task_name: str = 'lam_openx',
            optimizer: OptimizerCallable = AdamW,
            make_data_pair: bool = False,

    ) -> None:
        super(DINO_LAM, self).__init__()
        self.lam = ControllableDINOLatentActionModel(
                in_dim=image_channels,
                model_dim=lam_model_dim,
                latent_dim=lam_latent_dim,
                num_latents=lam_num_latents,
                patch_size=lam_patch_size,
                enc_blocks=lam_enc_blocks,
                dec_blocks=lam_dec_blocks,
                num_heads=lam_num_heads,
                dropout=lam_dropout,
            )

        self.lam_num_latents = lam_num_latents
        self.vq_beta = vq_beta
        self.log_interval = log_interval
        self.log_path = log_path
        self.optimizer = optimizer
        self.make_data_pair = make_data_pair

        self.save_hyperparameters()

        self.task_name = task_name
        self.distributed_state = PartialState()
        if self.distributed_state.is_main_process:
            wandb.init(name=task_name, reinit=True)

    def shared_step(self, batch: Dict) -> Tuple:
        # batch: keys['videos', 'task_instruction', 'action', 'dataset_names']

        outputs = self.lam(batch)
        gt_future_frames = outputs["target"]

        # Compute loss
        mse_loss = ((gt_future_frames - outputs["recon"]) ** 2).mean()
        q_loss = ((outputs["emb"].detach() - outputs["z"]) ** 2).mean()
        commit_loss = ((outputs["emb"] - outputs["z"].detach()) ** 2).mean()

        loss = mse_loss + q_loss + self.vq_beta * commit_loss
        
        # Optimize uncontrollable queries in stage-2
        if "z_q_uncontrol" in outputs.keys():
            q_loss_uncontrol = ((outputs["emb_uncontrol"].detach() - outputs["z_uncontrol"]) ** 2).mean()
            commit_loss_uncontrol = ((outputs["emb_uncontrol"]- outputs["z_uncontrol"].detach()) ** 2).mean()
            loss = loss + 0.1 * (q_loss_uncontrol + self.vq_beta * commit_loss_uncontrol)

        # Compute code usage
        unique, counts = torch.unique(outputs["indices"], return_counts=True)
        index_counts = torch.zeros(self.lam_num_latents, dtype=torch.long).cuda()
        index_counts[unique] = counts
        code_usage = (index_counts != 0).float().mean()

        loss_logs = (
            ("mse_loss", mse_loss),
            ("q_loss", q_loss),
            ("commit_loss", commit_loss),
            ("code_usage", code_usage),
        )

        if "indices_uncontrol" in outputs.keys():
            unique, counts = torch.unique(outputs["indices_uncontrol"], return_counts=True)
            index_counts = torch.zeros(32, dtype=torch.long).cuda()
            index_counts[unique] = counts
            uncontrol_code_usage = (index_counts != 0).float().mean()

            loss_logs = (
                ("mse_loss", mse_loss),
                ("q_loss", q_loss),
                ("commit_loss", commit_loss),
                ("q_loss_uncontrol", q_loss_uncontrol),
                ("commit_loss_uncontrol", commit_loss_uncontrol),
                ("code_usage", code_usage),
                ("code_usage_uncontrol", uncontrol_code_usage),
            )

        return outputs, loss, loss_logs



    def training_step(self, batch: Dict, batch_idx: int) -> Tensor:
        # Compute the training loss
        outputs, loss, aux_losses = self.shared_step(batch)


        # Log the training loss
        self.log_dict(
            {**{"train_loss": loss}, **{f"train/{k}": v for k, v in aux_losses}},
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True
        )

        if self.distributed_state.is_main_process:
            wandb.log({**{"train_loss": loss}, **{f"train/{k}": v for k, v in aux_losses}})

        # if batch_idx % self.log_interval == 0:  # Start of the epoch
        #     self.log_images(batch, outputs, "train")

        return loss

    # @torch.no_grad()
    # def validation_step(self, batch: Dict, batch_idx: int) -> Tensor:
    #     # Compute the validation loss
    #     outputs, loss, aux_losses = self.shared_step(batch)
    #
    #     # Log the validation loss
    #     self.log_dict(
    #         {**{"val_loss": loss}, **{f"val/{k}": v for k, v in aux_losses}},
    #         prog_bar=True,
    #         logger=True,
    #         on_step=True,
    #         on_epoch=True,
    #         sync_dist=True
    #     )
    #
    #     if batch_idx % self.log_interval == 0:  # Start of the epoch
    #         self.log_images(batch, outputs, "val")
    #     return loss

    @torch.no_grad()
    def test_step(self, batch: Dict, batch_idx: int) -> Tensor:
        # Compute the test loss
        outputs, loss, aux_losses = self.shared_step(batch)

        # Log the test loss
        self.log_dict(
            {**{"test_loss": loss}, **{f"test/{k}": v for k, v in aux_losses}},
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True
        )

        # self.log_images(batch, outputs, "test")

        return loss

    def log_images(self, batch: Dict, outputs: Dict, split: str) -> None:
        # gt_seq = batch["videos"][0][1:].clamp(0, 1).cpu()
        compare_seq = batch["videos"][0].clamp(0, 1).cpu()

        compare_seq = rearrange(compare_seq * 255, "t c h w -> h (t w) c")
        compare_seq = compare_seq.detach().numpy().astype(np.uint8)
        if split == 'test':
            import random
            idx = random.randint(0,100)
            img_path = path.join(self.log_path, f"vis_{self.task_name}", f"{split}_step{idx:06}.png")
        else:
            img_path = path.join(self.log_path, f"vis_{self.task_name}", f"{split}_step{self.global_step:06}.png")
        makedirs(path.dirname(img_path), exist_ok=True)
        img = Image.fromarray(compare_seq)
        img.save(img_path)

    def on_train_epoch_end(self):
        self.lam.vq.random_restart()
        self.lam.vq.reset_usage()

    def on_test_epoch_end(self):
        if self.make_data_pair:
            completed = len(listdir("output_pairs"))
            todo_name = listdir("../data/retro")[completed]
            makedirs(f"output_pairs/{todo_name}")
            top_indices = torch.topk(self.lam.vq.usage, 16, largest=True, sorted=True).indices
            top_latents = self.lam.vq.codebook(top_indices)
            torch.save(top_latents, f"output_pairs/{todo_name}/top_16.pt")
            with open(f"output_pairs/{todo_name}/top_16.txt", "w") as f:
                f.write(" ".join([str(i) for i in top_indices.tolist()]))

        self.plot_usage_distribution(self.lam.vq.usage, "unsorted_usage")
        self.plot_usage_distribution(self.lam.vq.usage.sort().values, "sorted_usage")

    def plot_usage_distribution(self, usage, filename):
        data = usage.cpu().numpy()
        n = 1
        for n in range(1, 10):
            if (2 ** n) ** 2 <= len(data) < (2 ** (n + 1)) ** 2:
                break
        data = data.reshape(2 ** n, -1)
        fig, ax = plt.subplots()
        cax = ax.matshow(data, interpolation="nearest")
        fig.colorbar(cax)
        plt.axis("off")
        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig(f"{filename}.png", bbox_inches="tight", pad_inches=0.0)
        plt.close()

    def configure_optimizers(self) -> Optimizer:
        optim = self.optimizer(self.parameters())
        return optim
