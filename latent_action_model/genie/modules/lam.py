from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange, repeat
from transformers import T5EncoderModel, T5Tokenizer

from latent_action_model.genie.modules.blocks import patchify, unpatchify, SpatioTemporalTransformer, SpatioTransformer, VectorQuantizer, \
                                                     MVSpatioTemporalTransformer, MVSpatioTransformer


class LatentActionModel(nn.Module):
    """
    Latent action VQ-VAE.
    """

    def __init__(
            self,
            in_dim: int,
            model_dim: int,
            latent_dim: int,
            num_latents: int,
            patch_size: int,
            enc_blocks: int,
            dec_blocks: int,
            num_heads: int,
            dropout: float = 0.0
    ) -> None:
        super(LatentActionModel, self).__init__()
        self.latent_dim = latent_dim
        self.patch_size = patch_size
        patch_token_dim = in_dim * patch_size ** 2

        self.num_codes = 4
        self.action_latent = nn.Parameter(torch.empty(1, 1, self.num_codes, patch_token_dim))    # TODO: num of codes
        nn.init.uniform_(self.action_latent, a=-1, b=1)
        self.encoder = SpatioTemporalTransformer(
            in_dim=patch_token_dim,
            model_dim=model_dim,
            out_dim=latent_dim,
            num_blocks=enc_blocks,
            num_heads=num_heads,
            dropout=dropout,
            causal_temporal=True
        )
        self.vq = VectorQuantizer(
            num_latents=num_latents,
            latent_dim=latent_dim,
            code_restart=True,
        )
        ## Decoder: Spatial Transformer
        self.patch_up = nn.Linear(patch_token_dim, model_dim)
        self.action_up = nn.Linear(latent_dim, model_dim)
        self.decoder = SpatioTransformer(
            in_dim=model_dim,
            model_dim=model_dim,
            out_dim=patch_token_dim,
            num_blocks=dec_blocks,
            num_heads=num_heads,
            dropout=dropout,
        )

    def vq_encode(self, videos: Tensor) -> Dict:
        # Preprocess videos
        B, T = videos.shape[:2]
        patches = patchify(videos, self.patch_size)
        action_pad = self.action_latent.expand(B, T, -1, -1)
        padded_patches = torch.cat([action_pad, patches], dim=2)

        # Encode
        z = self.encoder(padded_patches)  # (B, T, 1+N, E)
        # Get latent action for all future frames
        z = z[:, 1:, :self.num_codes]  # (B, T-1, n, E)

        # Vector quantize
        z = z.reshape(B * (T - 1), self.num_codes, self.latent_dim)
        z_q, z, emb, indices = self.vq(z)
        z_q = z_q.reshape(B, T - 1, self.num_codes, self.latent_dim)
        return {
            "patches": patches,
            "z_q": z_q,
            "z": z,
            "emb": emb,
            "indices": indices
        }

    def forward(self, batch: Dict) -> Dict:
        # Encode + VQ
        H, W = batch["videos"].shape[3:5]
        outputs = self.vq_encode(batch["videos"])
        video_patches = self.patch_up(outputs["patches"][:, :-1])
        action_patches = self.action_up(outputs["z_q"])
        video_action_patches = torch.cat([action_patches, video_patches], dim=2)

        del outputs["patches"]

        # Decode
        video_recon = self.decoder(video_action_patches)
        video_recon = video_recon[:, :, self.num_codes:] 
        video_recon = F.sigmoid(video_recon)

        outputs.update(
            {
                "recon": unpatchify(video_recon, self.patch_size, H, W)
            }
        )
        return outputs

    @property
    def device(self):
        return next(self.parameters()).device


class LatentActionModel_MultiView(nn.Module):
    """
    Latent action VQ-VAE.
    """

    def __init__(
            self,
            in_dim: int,
            model_dim: int,
            latent_dim: int,
            num_latents: int,
            patch_size: int,
            enc_blocks: int,
            dec_blocks: int,
            num_heads: int,
            dropout: float = 0.0
    ) -> None:
        super(LatentActionModel_MultiView, self).__init__()
        self.latent_dim = latent_dim
        self.patch_size = patch_size
        patch_token_dim = in_dim * patch_size ** 2

        self.num_codes = 4
        self.action_latent = nn.Parameter(torch.empty(1, 1, self.num_codes, model_dim))   
        nn.init.uniform_(self.action_latent, a=-1, b=1)
        self.encoder = MVSpatioTemporalTransformer(
            in_dim=patch_token_dim,
            model_dim=model_dim,
            out_dim=latent_dim,
            num_blocks=enc_blocks,
            num_heads=num_heads,
            dropout=dropout,
            causal_temporal=True
        )
        self.vq = VectorQuantizer(
            num_latents=num_latents,
            latent_dim=latent_dim,
            code_restart=True,
        )
        ## Decoder: Spatial Transformer
        self.patch_up = nn.Linear(patch_token_dim, model_dim)
        self.action_up_view1 = nn.Sequential(nn.Linear(latent_dim, model_dim),
                                                nn.GELU(),
                                                nn.Linear(model_dim, model_dim)
                                            )
        self.action_up_view2 = nn.Sequential(nn.Linear(latent_dim, model_dim),
                                                nn.GELU(),
                                                nn.Linear(model_dim, model_dim)
                                            )
        self.decoder = MVSpatioTransformer(
            in_dim=model_dim,
            model_dim=model_dim,
            out_dim=patch_token_dim,
            num_blocks=dec_blocks,
            num_heads=num_heads,
            dropout=dropout,
        )

    def vq_encode(self, videos: Tensor, videos_view2: Tensor) -> Dict:
        # Preprocess videos
        B, T = videos.shape[:2]
        patches = patchify(videos, self.patch_size)
        patches_view2 = patchify(videos_view2, self.patch_size)
        action_pad = self.action_latent.expand(B, T, -1, -1)
        # padded_patches = torch.cat([action_pad, patches, patches_view2], dim=2)

        # Encode
        z = self.encoder(action_pad, patches, patches_view2)  # (B, T, 1+N, E)
        # Get latent action for all future frames
        z = z[:, 1:, :self.num_codes]  # (B, T-1, n, E)

        # Vector quantize
        z = z.reshape(B * (T - 1), self.num_codes, self.latent_dim)
        z_q, z, emb, indices = self.vq(z)
        z_q = z_q.reshape(B, T - 1, self.num_codes, self.latent_dim)
        return {
            "patches": patches,
            "patches_view2": patches_view2,
            "z_q": z_q,
            "z": z,
            "emb": emb,
            "indices": indices
        }

    def forward(self, batch: Dict) -> Dict:
        # Encode + VQ
        H, W = batch["videos"].shape[3:5]
        outputs = self.vq_encode(batch["videos"], batch['videos_view2'])
        video_patches = self.patch_up(outputs["patches"][:, :-1])
        video_patches_view2 = self.patch_up(outputs["patches_view2"][:, :-1])
        # action_patches = self.action_up(outputs["z_q"])
        # video_action_patches = torch.cat([action_patches, video_patches], dim=2)

        del outputs["patches"]

        # Decode
        video_recon_view1 = self.decoder(self.action_up_view1(outputs["z_q"]), video_patches)
        video_recon_view1 = video_recon_view1[:, :, self.num_codes:] 
        video_recon_view1 = F.sigmoid(video_recon_view1)

        video_recon_view2 = self.decoder(self.action_up_view2(outputs["z_q"]), video_patches_view2)
        video_recon_view2 = video_recon_view2[:, :, self.num_codes:] 
        video_recon_view2 = F.sigmoid(video_recon_view2)
        
        # video_recon_view1 = video_recon[:,:, :video_patches.shape[2]]
        # video_recon_view2 = video_recon[:,:, video_patches.shape[2]:]

        outputs.update(
            {
                "recon": unpatchify(video_recon_view1, self.patch_size, H, W),
                "recon_view2": unpatchify(video_recon_view2, self.patch_size, H, W)
            }
        )
        return outputs

    @property
    def device(self):
        return next(self.parameters()).device


class ControlAwareLatentActionModel(nn.Module):
    """
    Latent action VQ-VAE.
    """

    def __init__(
            self,
            in_dim: int,
            model_dim: int,
            latent_dim: int,
            num_latents: int,
            patch_size: int,
            enc_blocks: int,
            dec_blocks: int,
            num_heads: int,
            dropout: float = 0.0,
            stage_2: bool = False,
    ) -> None:
        super(ControlAwareLatentActionModel, self).__init__()
        self.latent_dim = latent_dim
        self.patch_size = patch_size
        patch_token_dim = in_dim * patch_size ** 2

        self.num_codes = 4
        self.action_latent = nn.Parameter(torch.empty(1, 1, self.num_codes, patch_token_dim))
        nn.init.uniform_(self.action_latent, a=-1, b=1)
        self.encoder = SpatioTemporalTransformer(
            in_dim=patch_token_dim,
            model_dim=model_dim,
            out_dim=latent_dim,
            num_blocks=enc_blocks,
            num_heads=num_heads,
            dropout=dropout,
            causal_temporal=True,
            to_out=False,
        )

        self.to_codebook = nn.Linear(model_dim, latent_dim)
        self.vq = VectorQuantizer(
            num_latents=num_latents,
            latent_dim=latent_dim,
            code_restart=True,
        )
        if stage_2:
            self.vq_action = VectorQuantizer(
                num_latents=num_latents,
                latent_dim=latent_dim,
                code_restart=True,
            )
            self.action_latent_controllable = nn.Parameter(torch.empty(1, 1, self.num_codes, patch_token_dim))
            nn.init.uniform_(self.action_latent_controllable, a=-1, b=1)

            self.vq.requires_grad_(False)
            # self.action_latent.requires_grad_(False)


        ## Decoder: Spatial Transformer
        self.patch_up = nn.Linear(patch_token_dim, model_dim)
        self.action_up = nn.Linear(latent_dim, model_dim)
        self.decoder = SpatioTransformer(
            in_dim=model_dim,
            model_dim=model_dim,
            out_dim=patch_token_dim,
            num_blocks=dec_blocks,
            num_heads=num_heads,
            dropout=dropout,
        )

        if not stage_2:
            # Load T5 text encoder model
            self.text_encoder = T5EncoderModel.from_pretrained('/cpfs01/user/buqingwen/t5-small')
            self.text_encoder.requires_grad_(False)
            self.lang_proj = nn.Linear(512, model_dim)

            # Load T5 tokenizer
            self.tokenizer = T5Tokenizer.from_pretrained('/cpfs01/user/buqingwen/t5-small')

        self.stage_2 = stage_2


    def encode_text(self, lang: List):
        # Tokenize the batch with padding to the longest sequence
        encoding = self.tokenizer(lang, return_tensors="pt", padding=True).to(self.device) 

        # Access the input IDs and attention masks
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']

        # Get encoder outputs
        with torch.no_grad():
            encoder_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)

        # Access the last hidden states
        last_hidden_states = encoder_outputs.last_hidden_state

        return last_hidden_states, attention_mask


    def vq_encode(self, videos: Tensor, lang_embed: Tensor = None, attention_mask: Tensor = None) -> Dict:
        # Preprocess videos
        B, T = videos.shape[:2]
        patches = patchify(videos, self.patch_size)
        action_pad = self.action_latent.expand(B, T, -1, -1)

        # Action latents as queries
        padded_patches = torch.cat([action_pad, patches], dim=2)

        num_latent_actions = self.num_codes
        if self.stage_2:
            num_latent_actions *= 2
            action_pad_controllable = self.action_latent_controllable.expand(B, T, -1, -1)
            padded_patches = torch.cat([action_pad_controllable, padded_patches], dim=2)
        # Encode
        z = self.encoder(padded_patches, lang_embed, attention_mask)  # (B, T, 1+N, E)

        # Get latent action for all future frames
        if self.stage_2:
            latent_action = self.to_codebook(z[:, 1:, self.num_codes: self.num_codes * 2])
            latent_action_controllable = self.to_codebook(z[:, 1:, :self.num_codes])
        else:
            latent_action = self.to_codebook(z[:, 1:, :self.num_codes])  # (B, T-1, n, E)


        # Get action-irrelevant shortcut embeddings
        shortcut_emb = z[:, 1:, num_latent_actions: num_latent_actions + patches.shape[2]]

        # Get action-grounded lang embedding
        if not self.stage_2:
            lang_emb = z[:, 1:, num_latent_actions + patches.shape[2]:]

        # Vector quantize
        latent_action = latent_action.reshape(B * (T - 1), self.num_codes, self.latent_dim)
        z_q, z, emb, indices = self.vq(latent_action)
        z_q = z_q.reshape(B, T - 1, self.num_codes, self.latent_dim)

        if self.stage_2:
            latent_action_controllable = latent_action_controllable.reshape(B * (T - 1), self.num_codes, self.latent_dim)
            z_q_action, z_action, emb_action, indices = self.vq_action(latent_action_controllable)
            z_q_action = z_q_action.reshape(B, T - 1, self.num_codes, self.latent_dim)

        return {
            "patches": patches,
            "z_q": z_q,
            "z_q_action": z_q_action if self.stage_2 else None,
            "z": z,
            "z_action": z_action if self.stage_2 else None,
            "emb": emb,
            "emb_action": emb_action if self.stage_2 else None,
            "lang_emb": lang_emb if not self.stage_2 else None,
            "indices": indices,
            "shortcut": shortcut_emb,
        }

    def forward(self, batch: Dict) -> Dict:
        B, T = batch["videos"].shape[:2]
        H, W = batch["videos"].shape[3:5]

        num_latent_actions = self.num_codes if not self.stage_2 else self.num_codes * 2
        # Encode task instructions
        if not self.stage_2:
            lang_embed, attention_mask = self.encode_text(batch["task_instruction"])
            lang_embed = self.lang_proj(lang_embed)
            attention_mask = torch.cat([torch.ones((B, num_latent_actions + (H // self.patch_size)**2)).to(self.device),
                                        attention_mask],
                                        dim = -1)

        # Encode
        if self.stage_2:
            outputs = self.vq_encode(batch["videos"]) #, repeat(lang_embed, 'b l d -> b T l d', T=T), attention_mask.repeat(T, 1)) 
        else:
            outputs = self.vq_encode(batch["videos"], repeat(lang_embed, 'b l d -> b T l d', T=T), attention_mask.repeat(T, 1)) 
        video_patches = self.patch_up(outputs["patches"][:, :-1]) #+ outputs['shortcut']
        action_patches = self.action_up(outputs["z_q"])
        video_action_patches = torch.cat([action_patches, video_patches], dim=2)

        del outputs["patches"]

        # Decode
        
        if not self.stage_2:
            video_recon = self.decoder(video_action_patches, outputs["lang_emb"], attention_mask)
        else:
            controllable_action_patches = self.action_up(outputs["z_q_action"])
            video_action_patches = torch.cat([controllable_action_patches, video_action_patches], dim=2)
            video_recon = self.decoder(video_action_patches)

        video_recon = video_recon[:, :, num_latent_actions: num_latent_actions + (H // self.patch_size)**2 ] 
        video_recon = F.sigmoid(video_recon)

        outputs.update(
            {
                "recon": unpatchify(video_recon, self.patch_size, H, W)
            }
        )
        return outputs

    @property
    def device(self):
        return next(self.parameters()).device

from torchvision import transforms
# Use timm's names
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class DINOLatentActionModel(nn.Module):
    """
    Latent action VQ-VAE.
    """

    def __init__(
            self,
            in_dim: int,
            model_dim: int,
            latent_dim: int,
            num_latents: int,
            patch_size: int,
            enc_blocks: int,
            dec_blocks: int,
            num_heads: int,
            dropout: float = 0.0
    ) -> None:
        super(DINOLatentActionModel, self).__init__()
        self.latent_dim = latent_dim
        self.patch_size = patch_size
        patch_token_dim = in_dim * patch_size ** 2

        self.dino_transform = transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
        self.dino_encoder = torch.hub.load('/cpfs01/shared/opendrivelab/chenjin/facebookresearch_dinov2_main', 'dinov2_vitb14_reg',
                                          source='local', pretrained=False)
        self.dino_encoder.load_state_dict(torch.load('/cpfs01/shared/opendrivelab/chenjin/facebookresearch_dinov2_main/dinov2_vitb14_reg4_pretrain.pth', map_location='cpu'))
        self.dino_encoder.requires_grad_(False)

        dino_dim = 768

        self.num_codes = 4
        self.action_latent = nn.Parameter(torch.empty(1, 1, self.num_codes, dino_dim))    # TODO: num of codes
        nn.init.uniform_(self.action_latent, a=-1, b=1)
        self.encoder = SpatioTemporalTransformer(
            in_dim=dino_dim,
            model_dim=model_dim,
            out_dim=latent_dim,
            num_blocks=enc_blocks,
            num_heads=num_heads,
            dropout=dropout,
            causal_temporal=True
        )
        self.vq = VectorQuantizer(
            num_latents=num_latents,
            latent_dim=latent_dim,
            code_restart=True,
        )
        ## Decoder: Spatial Transformer
        self.patch_up = nn.Linear(dino_dim, model_dim)
        self.action_up = nn.Linear(latent_dim, model_dim)
        self.decoder = SpatioTransformer(
            in_dim=model_dim,
            model_dim=model_dim,
            out_dim=dino_dim,        # Dim of DINOv2-Base
            num_blocks=dec_blocks,
            num_heads=num_heads,
            dropout=dropout,
        )

    def vq_encode(self, videos: Tensor) -> Dict:
        # Preprocess videos
        B, T = videos.shape[:2]
        videos = rearrange(videos, "b T c h w -> (b T) c h w")
        videos = self.dino_transform(videos)
        dion_features = self.dino_encoder.forward_features(videos)['x_norm_patchtokens']
        dion_features = rearrange(dion_features, "(b T) l d -> b T l d", T=2)

        action_pad = self.action_latent.expand(B, T, -1, -1)
        padded_patches = torch.cat([action_pad, dion_features], dim=2)

        # Encode
        z = self.encoder(padded_patches)  # (B, T, 1+N, E)
        # Get latent action for all future frames
        z = z[:, 1:, :self.num_codes]  # (B, T-1, n, E)

        # Vector quantize
        z = z.reshape(B * (T - 1), self.num_codes, self.latent_dim)
        z_q, z, emb, indices = self.vq(z)
        z_q = z_q.reshape(B, T - 1, self.num_codes, self.latent_dim)
        return {
            "patches": dion_features,
            "z_q": z_q,
            "z": z,
            "emb": emb,
            "indices": indices
        }

    def forward(self, batch: Dict) -> Dict:
        # Encode + VQ
        H, W = batch["videos"].shape[3:5]
        outputs = self.vq_encode(batch["videos"])
        video_patches = self.patch_up(outputs["patches"][:, :-1])
        action_patches = self.action_up(outputs["z_q"])
        video_action_patches = torch.cat([action_patches, video_patches], dim=2)

        # Decode
        video_recon = self.decoder(video_action_patches)
        video_recon = video_recon[:, :, self.num_codes:] 

        outputs.update(
            {
                "recon": video_recon,
                "target": outputs["patches"][:, [-1]]
            }
        )
        return outputs

    @property
    def device(self):
        return next(self.parameters()).device



class UncontrolledDINOLatentActionModel(nn.Module):
    """
    Latent action VQ-VAE.
    """

    def __init__(
            self,
            in_dim: int,
            model_dim: int,
            latent_dim: int,
            num_latents: int,
            patch_size: int,
            enc_blocks: int,
            dec_blocks: int,
            num_heads: int,
            dropout: float = 0.0
    ) -> None:
        super(UncontrolledDINOLatentActionModel, self).__init__()
        self.latent_dim = latent_dim
        self.patch_size = patch_size
        patch_token_dim = in_dim * patch_size ** 2

        self.dino_transform = transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
        self.dino_encoder = torch.hub.load('/cpfs01/user/buqingwen/facebookresearch_dinov2_main', 'dinov2_vitb14_reg',
                                          source='local', pretrained=False)
        self.dino_encoder.load_state_dict(torch.load('/cpfs01/user/buqingwen/dinov2_vitb14_reg4_pretrain.pth', map_location='cpu'))
        self.dino_encoder.requires_grad_(False)

        dino_dim = 768

        self.num_codes = 4
        self.action_latent = nn.Parameter(torch.empty(1, 1, self.num_codes, dino_dim))    # TODO: num of codes
        nn.init.uniform_(self.action_latent, a=-1, b=1)
        self.encoder = SpatioTemporalTransformer(
            in_dim=dino_dim,
            model_dim=model_dim,
            out_dim=latent_dim,
            num_blocks=enc_blocks,
            num_heads=num_heads,
            dropout=dropout,
            causal_temporal=True,
            to_out=False,
        )

        self.to_codebook = nn.Linear(model_dim, latent_dim)
        self.vq = VectorQuantizer(
            num_latents=num_latents,
            latent_dim=latent_dim,
            code_restart=True,
        )
        ## Decoder: Spatial Transformer
        self.patch_up = nn.Linear(dino_dim, model_dim)
        self.action_up = nn.Linear(latent_dim, model_dim)
        self.decoder = SpatioTransformer(
            in_dim=model_dim,
            model_dim=model_dim,
            out_dim=dino_dim,        # Dim of DINOv2-Base
            num_blocks=dec_blocks,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Load T5 text encoder model
        self.text_encoder = T5EncoderModel.from_pretrained('/cpfs01/user/buqingwen/t5-small')
        self.text_encoder.requires_grad_(False)
        self.lang_proj = nn.Linear(512, model_dim)

        # Load T5 tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained('/cpfs01/user/buqingwen/t5-small')

    def encode_text(self, lang: List):
        # Tokenize the batch with padding to the longest sequence
        encoding = self.tokenizer(lang, return_tensors="pt", padding=True).to(self.device) 

        # Access the input IDs and attention masks
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']

        # Get encoder outputs
        with torch.no_grad():
            encoder_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)

        # Access the last hidden states
        last_hidden_states = encoder_outputs.last_hidden_state

        return last_hidden_states, attention_mask

    def vq_encode(self, videos: Tensor, lang_embed: Tensor = None, attention_mask: Tensor = None) -> Dict:
        # Preprocess videos
        B, T = videos.shape[:2]
        videos = rearrange(videos, "b T c h w -> (b T) c h w")
        videos = self.dino_transform(videos)
        dion_features = self.dino_encoder.forward_features(videos)['x_norm_patchtokens']
        dion_features = rearrange(dion_features, "(b T) l d -> b T l d", T=2)

        action_pad = self.action_latent.expand(B, T, -1, -1)
        padded_patches = torch.cat([action_pad, dion_features], dim=2)

        # Encode
        z = self.encoder(padded_patches, lang_embed, attention_mask) 

        # Get language embedding
        lang_emb = z[:, 1:, self.num_codes + dion_features.shape[2]:]

        # Get latent action for all future frames
        z = self.to_codebook(z[:, 1:, :self.num_codes])  # (B, T-1, n, E)

        

        # Vector quantize
        z = z.reshape(B * (T - 1), self.num_codes, self.latent_dim)
        z_q, z, emb, indices = self.vq(z)
        z_q = z_q.reshape(B, T - 1, self.num_codes, self.latent_dim)
        return {
            "patches": dion_features,
            "z_q": z_q,
            "z": z,
            "emb": emb,
            "indices": indices,
            "lang_emb": lang_emb
        }

    def forward(self, batch: Dict) -> Dict:
        # Encode + VQ
        B, T = batch["videos"].shape[:2]
        H, W = batch["videos"].shape[3:5]

        lang_embed, attention_mask = self.encode_text(batch["task_instruction"])
        lang_embed = self.lang_proj(lang_embed)
        attention_mask = torch.cat([torch.ones((B, self.num_codes + (H // self.patch_size)**2)).to(self.device),
                                    attention_mask],
                                    dim = -1)

        outputs = self.vq_encode(batch["videos"], repeat(lang_embed, 'b l d -> b T l d', T=T), attention_mask.repeat(T, 1)) 
        video_patches = self.patch_up(outputs["patches"][:, :-1])
        action_patches = self.action_up(outputs["z_q"])
        video_action_patches = torch.cat([action_patches, video_patches], dim=2)

        # Decode
        video_recon = self.decoder(video_action_patches, outputs["lang_emb"], attention_mask)
        video_recon = video_recon[:, :, self.num_codes: self.num_codes + video_patches.shape[2]] 

        outputs.update(
            {
                "recon": video_recon,
                "target": outputs["patches"][:, [-1]]
            }
        )
        return outputs

    @property
    def device(self):
        return next(self.parameters()).device




class ControllableDINOLatentActionModel(nn.Module):
    """
    Latent action VQ-VAE.
    """

    def __init__(
            self,
            in_dim: int,
            model_dim: int,
            latent_dim: int,
            num_latents: int,
            patch_size: int,
            enc_blocks: int,
            dec_blocks: int,
            num_heads: int,
            dropout: float = 0.0
    ) -> None:
        super(ControllableDINOLatentActionModel, self).__init__()
        self.latent_dim = latent_dim
        self.patch_size = patch_size
        patch_token_dim = in_dim * patch_size ** 2

        self.dino_transform = transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
        self.dino_encoder = torch.hub.load('/cpfs01/shared/opendrivelab/qwbu/facebookresearch_dinov2_main', 'dinov2_vitb14_reg',
                                          source='local', pretrained=False)
        self.dino_encoder.load_state_dict(torch.load('/cpfs01/shared/opendrivelab/qwbu/dinov2_vitb14_reg4_pretrain.pth', map_location='cpu'))
        self.dino_encoder.requires_grad_(False)

        dino_dim = 768

        self.num_codes = 4
        self.action_latent = nn.Parameter(torch.empty(1, 1, self.num_codes, dino_dim))    # TODO: num of codes
        nn.init.uniform_(self.action_latent, a=-1, b=1)
        self.encoder = SpatioTemporalTransformer(
            in_dim=dino_dim,
            model_dim=model_dim,
            out_dim=latent_dim,
            num_blocks=enc_blocks,
            num_heads=num_heads,
            dropout=dropout,
            causal_temporal=True,
            to_out=False,
        )

        self.to_codebook = nn.Linear(model_dim, latent_dim)
        self.to_codebook_uncontrol = nn.Linear(model_dim, latent_dim)
        self.vq = VectorQuantizer(
            num_latents=num_latents,
            latent_dim=latent_dim,
            code_restart=True,
        )
        ## Decoder: Spatial Transformer
        self.patch_up = nn.Linear(dino_dim, model_dim)
        self.action_up = nn.Linear(latent_dim, model_dim)
        self.action_up_uncontrol = nn.Linear(latent_dim, model_dim)
        self.decoder = SpatioTransformer(
            in_dim=model_dim,
            model_dim=model_dim,
            out_dim=dino_dim,        # Dim of DINOv2-Base
            num_blocks=dec_blocks,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.vq_action = VectorQuantizer(
                num_latents=num_latents,
                latent_dim=latent_dim,
                code_restart=True,
            )
        self.action_latent_controllable = nn.Parameter(torch.empty(1, 1, self.num_codes, dino_dim))
        nn.init.uniform_(self.action_latent_controllable, a=-1, b=1)

        # Only optimize the new codebook
        # self.vq.requires_grad_(False)
        # self.action_latent.requires_grad_(False)


    def vq_encode(self, videos: Tensor, lang_embed: Tensor = None, attention_mask: Tensor = None) -> Dict:
        # Preprocess videos
        B, T = videos.shape[:2]
        videos = rearrange(videos, "b T c h w -> (b T) c h w")
        videos = self.dino_transform(videos)
        dion_features = self.dino_encoder.forward_features(videos)['x_norm_patchtokens']
        dion_features = rearrange(dion_features, "(b T) l d -> b T l d", T=2)

        
        action_pad = self.action_latent.expand(B, T, -1, -1)
        padded_patches = torch.cat([action_pad, dion_features], dim=2)

        action_pad_controllable = self.action_latent_controllable.expand(B, T, -1, -1)
        padded_patches = torch.cat([action_pad_controllable, padded_patches], dim=2)

        # Encode
        z = self.encoder(padded_patches) 


        # Get 'uncotrollable' latent action for all future frames
        z_uncontrol = self.to_codebook_uncontrol(z[:, 1:, self.num_codes : self.num_codes * 2])

        # Vector quantize
        z_uncontrol = z_uncontrol.reshape(B * (T - 1), self.num_codes, self.latent_dim)
        z_q_uncontrol, z_uncontrol, emb_uncontrol, indices_uncontrol = self.vq(z_uncontrol)
        z_q_uncontrol = z_q_uncontrol.reshape(B, T - 1, self.num_codes, self.latent_dim)


        # Get 'cotrollable' latent action for all future frames
        z_action = self.to_codebook(z[:, 1:, :self.num_codes])  # (B, T-1, n, E)

        # Vector quantize
        z_action = z_action.reshape(B * (T - 1), self.num_codes, self.latent_dim)
        z_q, z, emb, indices = self.vq_action(z_action)
        z_q = z_q.reshape(B, T - 1, self.num_codes, self.latent_dim)



        return {
            "patches": dion_features,
            "z_q": z_q,
            "z": z,
            "emb": emb,
            "z_q_uncontrol": z_q_uncontrol,
            "z_uncontrol": z_uncontrol,
            "emb_uncontrol": emb_uncontrol,
            "indices": indices,
            "indices_uncontrol": indices_uncontrol,
        }

    def forward(self, batch: Dict) -> Dict:
        # Encode + VQ
        B, T = batch["videos"].shape[:2]
        H, W = batch["videos"].shape[3:5]

        outputs = self.vq_encode(batch["videos"]) 
        video_patches = self.patch_up(outputs["patches"][:, :-1])

        # Decode
        video_action_patches = torch.cat([self.action_up(outputs["z_q"]), 
                                          self.action_up_uncontrol(outputs["z_q_uncontrol"]), 
                                          video_patches],
                                          dim=2)
        video_recon = self.decoder(video_action_patches)
        video_recon = video_recon[:, :, -video_patches.shape[2]:] 

        outputs.update(
            {
                "recon": video_recon,
                "target": outputs["patches"][:, [-1]]
            }
        )
        return outputs

    @property
    def device(self):
        return next(self.parameters()).device