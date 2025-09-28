# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn
import torch.nn.functional as F
from mmdet3d.models.builder import HEADS, build_loss
from mmcv.runner import BaseModule, ModuleList, force_fp32
from mmcv.cnn import ConvModule, build_conv_layer, build_norm_layer, build_upsample_layer
from model.diffusers.schedulers.scheduling_ddim import DDIMScheduler

from typing import Union, Dict, Tuple, Optional
from .mmbev_base_depth_refine import BaseDepthRefine
from model.ops.depth_transform import DEPTH_TRANSFORM
from model.necks.hahi import HAHIHeteroNeck
from MinkowskiEngine import SparseTensor
import os
import logging
import numpy as np
from model.minknet import get_model as get_mink_model
from model.utils.fourier_embedding import FourierEmbedder
from model.utils.transformer_temporal import TemporalResnetBlock, TransformerSpatioTemporalModel, SpatioTemporalResBlock
import jax.numpy as jnp
from model.utils.embeddings import TimestepEmbedding, Timesteps

from model.utils.multi_head_attention_layer import MultiHeadAttentionLayer, DecoderBlock, Decoder, \
    PositionWiseFeedForwardLayer, TransformerEmbedding

from perceiver_pytorch import PerceiverIO

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


@HEADS.register_module()
class Video_Diffusion(BaseDepthRefine):
    def __init__(
            self,
            up_scale_factor=1,
            inference_steps=20,
            num_train_timesteps=1000,
            return_indices=None,
            mink_pretrained_path=None,
            video_fps=10,
            diff_type=None,
            image_width=None,
            image_height=None,
            image_model_path=None,
            ap_loc=True,
            cross_attention=True,
            **kwargs
    ):
        super().__init__(blur_depth_head=False, **kwargs)
        # channels_in = kwargs['in_channels'][0] + self.depth_embed_dim
        self.video_fps = video_fps
        self.mink_final_width = int(image_width / 2)
        self.mink_final_height = int(image_height / 2)

        self.ap_loc = ap_loc
        overshot_dim = 8 + 16 + 64
        pre_measured_dim = 8 + 16 + 64
        self.cross_attention_dim = 11
        self.video_cross_attention_dim = 8

        self.mink_final_dim = 1 + 1
        self.mesh_dim = 1
        refined_channels_in = overshot_dim + self.cross_attention_dim + pre_measured_dim + self.mink_final_dim + self.mesh_dim
        refined_channels_in += 1  # coord
        refined_channels_in += 1  # ap
        refined_channels_in += 8  # whole frame cross attention

        # 5+8+46+1+4 = 64
        in_channels = [192, 384, 768, 1536]
        # in_channels_3d = [96, 96, 128, 128]
        in_channels_3d = [1, 1, 1]

        # self.depth_feature_dim = kwargs['depth_feature_dim'] = 256
        self.depth_feature_dim = 16
        # self.interpolate_dim = (64, 10000)
        self.interpolate_dim = (64, 30000)
        # self.interpolate_dim = (64, 20000)

        # self.interpolate_dim2 = (1000, 48)
        self.interpolate_dim2 = (3000, 48)
        # self.interpolate_dim2 = (2000, 48)

        if up_scale_factor == 1:
            self.up_scale = nn.Identity()
        else:
            self.up_scale = lambda tensor: F.interpolate(tensor, scale_factor=up_scale_factor, mode='bilinear')

        # Minknet params
        classes = 21
        self.mink_model = get_mink_model("mink_unet", classes, multi_scale=True).cuda()
        if mink_pretrained_path:
            assert os.path.isfile(mink_pretrained_path), "The pretrained mink model doesn't exist" + str(
                mink_pretrained_path)
            logging.info("Load Pretrained MinkUnet")
            checkpoint = torch.load(mink_pretrained_path, map_location=lambda storage, loc: storage.cuda())
            try:
                self.mink_model.load_state_dict(checkpoint['state_dict'], strict=True)
            except RuntimeError:
                logging.error("Failed to Load Pretrained MinkUnet, but try the best!")
                self.mink_model.load_state_dict(checkpoint['state_dict'], strict=False)
                logging.info("Succeed to Load Pretrained MinkUnet!!")

        self.fourier_freqs = 8
        self.fourier_embedder = FourierEmbedder(num_freqs=self.fourier_freqs, temperature=2)
        # self.fourier_embedder2 = FourierEmbedder(num_freqs=self.fourier_freqs, temperature=2)

        position_dim = self.fourier_freqs * 2 * 6  # 2 is sin&cos, 6 is xyzxyz

        AP_loc_inner_features = 256
        self.coords_embedding_layer = nn.Sequential(
            nn.Linear(position_dim, AP_loc_inner_features, device='cuda'),
            nn.SiLU(),
            nn.Linear(AP_loc_inner_features, AP_loc_inner_features, device='cuda'),
            nn.SiLU(),
            nn.Linear(AP_loc_inner_features, 1 * self.mink_final_height * self.mink_final_width, device='cuda'),
        )
        multi_freq = self.fourier_freqs * 2
        freq_inner_features = 128
        self.freq_embedding_layer = nn.Sequential(
            nn.Linear(multi_freq, freq_inner_features, device='cuda'),
            nn.SiLU(),
            nn.Linear(freq_inner_features, freq_inner_features, device='cuda'),
            nn.SiLU(),
            nn.Linear(freq_inner_features, 1 * self.mink_final_height * self.mink_final_width, device='cuda'),
        )
        """
        point_inner_features = 256
        self.point_embedding_layer = nn.Sequential(
            nn.Linear((2 + 1) * self.pre_measured_num, point_inner_features, device='cuda'),
            nn.SiLU(),
            nn.Linear(point_inner_features, point_inner_features, device='cuda'),
            nn.SiLU(),
            nn.Linear(point_inner_features, 1 * self.mink_final_height * self.mink_final_width, device='cuda'),
        )
        """

        mink_inner_features = 128
        mink2_inner_features = 128

        self.mink_embedding_layer = nn.Sequential(
            nn.Linear(self.interpolate_dim[0] * self.interpolate_dim[1], mink_inner_features, device='cuda'),
            nn.ReLU(),
            nn.Linear(mink_inner_features, mink_inner_features, device='cuda'),
            nn.ReLU(),
            nn.Linear(mink_inner_features, self.mink_final_dim * self.interpolate_dim[0] * self.interpolate_dim[1],
                      device='cuda'),
        )
        self.mink_embedding_layer2 = nn.Sequential(
            nn.Linear(self.interpolate_dim2[0] * self.interpolate_dim2[1], mink2_inner_features, device='cuda'),
            nn.SiLU(),
            nn.Linear(mink2_inner_features, mink2_inner_features, device='cuda'),
            nn.SiLU(),
            nn.Linear(mink2_inner_features, self.mesh_dim * self.mink_final_height * self.mink_final_width, device='cuda'),
        )
        depth_transform_cfg = dict(type='DeepDepthTransformWithUpsamplingHeatmap_1D', hidden=self.depth_feature_dim,
                                   eps=1e-6)
        self.depth_transform = DEPTH_TRANSFORM.build(depth_transform_cfg)
        self.return_indices = return_indices
        self.model = ScheduledCNNRefine(channels_in=refined_channels_in, channels_noise=self.depth_feature_dim,
                                        video_fps=self.video_fps)

        # Load pretrained image model if exits
        if image_model_path:
            # Freeze image-diffuse related parts
            for param in self.model.noise_embedding.parameters():
                param.requires_grad = False
            for param in self.model.upsample_fuse.parameters():
                param.requires_grad = False
            for param in self.model.pred.parameters():
                param.requires_grad = False
            for param in self.model.time_embedding.parameters():
                param.requires_grad = False

        self.diffusion_inference_steps = inference_steps
        if diff_type == 'DDIM':
            self.scheduler = DDIMScheduler(num_train_timesteps=num_train_timesteps, clip_sample=False)
            self.pipeline = CNNDDIMPipiline(self.model, self.scheduler)

        del self.weight_head
        del self.conv_lateral
        del self.conv_up

        upsample_cfg = dict(type='deconv', bias=False)
        self.hahineck = HAHIHeteroNeck(in_channels=in_channels, out_channels=in_channels,
                                       embedding_dim=512,
                                       positional_encoding=dict(type='SinePositionalEncoding', num_feats=256),
                                       scales=[1, 1, 1, 1], cross_att=False, self_att=False, num_points=8).cuda()

        self.decoder_dim = self.mink_final_width * self.mink_final_height
        self.video_decoder_dim = 5000

        self.perceiver_model_mink = PerceiverIO(
            dim=96,  # dimension of sequence to be encoded
            queries_dim=self.decoder_dim,  # dimension of decoder queries
            # logits_dim=708,  # 177*88/22=708, +2 for just fitting into 177 x 88
            logits_dim=None,  # Make them return the feature not final logits
            depth=6,  # depth of net
            num_latents=256,
            # number of latents, or induced set points, or centroids. different papers giving it different names
            latent_dim=512,  # latent dimension
            cross_heads=8,  # number of heads for cross attention. paper said 1
            latent_heads=8,  # number of heads for latent self attention, 8
            cross_dim_head=self.cross_attention_dim,  # number of dimensions per cross attention head
            latent_dim_head=self.cross_attention_dim,  # number of dimensions per latent self attention head
            weight_tie_layers=False,  # whether to weight tie layers (optional, as indicated in the diagram)
            seq_dropout_prob=0.2
            # fraction of the tokens from the input sequence to dropout (structured dropout, for saving compute and regularizing effects)
        ).cuda()
        self.perceiver_model_video = PerceiverIO(
            dim=96,  # dimension of sequence to be encoded
            queries_dim=self.decoder_dim,  # dimension of decoder queries
            # logits_dim=708,  # 177*88/22=708, +2 for just fitting into 177 x 88
            logits_dim=None,  # Make them return the feature not final logits
            depth=6,  # depth of net
            num_latents=256,
            # number of latents, or induced set points, or centroids. different papers giving it different names
            latent_dim=512,  # latent dimension
            cross_heads=8,  # number of heads for cross attention. paper said 1
            latent_heads=8,  # number of heads for latent self attention, 8
            cross_dim_head=self.cross_attention_dim,  # number of dimensions per cross attention head
            latent_dim_head=self.cross_attention_dim,  # number of dimensions per latent self attention head
            weight_tie_layers=False,  # whether to weight tie layers (optional, as indicated in the diagram)
            seq_dropout_prob=0.2
            # fraction of the tokens from the input sequence to dropout (structured dropout, for saving compute and regularizing effects)
        ).cuda()
        # self.queries = torch.rand(self.cross_attention_dim, decoder_dim).cuda()
        # decoder(self.tgt_embed(tgt), encoder_out)

        self.conv_lateral = ModuleList()
        self.conv_up = ModuleList()
        self.conv_lateral2 = ModuleList()
        self.conv_up2 = ModuleList()
        self.conv_lateral_3d = ModuleList()
        self.conv_up_3d = ModuleList()
        for i in range(len(in_channels)):
            self.conv_lateral.append(
                nn.Sequential(
                    nn.Conv2d(in_channels[i], overshot_dim, 3, 1, 1, bias=False, device='cuda'),
                    build_norm_layer(dict(type='BN'), overshot_dim)[1],
                    nn.ReLU(True),
                    #     nn.Conv2d(depth_embed_dim, depth_embed_dim, 3, 1, 1, bias=False),
                    #     build_norm_layer(norm_cfg, depth_embed_dim)[1],
                    #     nn.ReLU(True),
                )
            )

            if i != 0:
                self.conv_up.append(
                    nn.Sequential(
                        build_upsample_layer(
                            upsample_cfg,
                            in_channels=overshot_dim,
                            out_channels=overshot_dim,
                            kernel_size=2,
                            stride=2,
                        ),
                        build_norm_layer(dict(type='BN'), overshot_dim)[1],
                        nn.ReLU(True),
                    )
                )

        for i in range(len(in_channels)):
            self.conv_lateral2.append(
                nn.Sequential(
                    nn.Conv2d(in_channels[i], pre_measured_dim, 3, 1, 1, bias=False, device='cuda'),
                    build_norm_layer(dict(type='BN'), pre_measured_dim)[1],
                    nn.ReLU(True),
                    #     nn.Conv2d(depth_embed_dim, depth_embed_dim, 3, 1, 1, bias=False),
                    #     build_norm_layer(norm_cfg, depth_embed_dim)[1],
                    #     nn.ReLU(True),
                )
            )

            if i != 0:
                self.conv_up2.append(
                    nn.Sequential(
                        build_upsample_layer(
                            upsample_cfg,
                            in_channels=pre_measured_dim,
                            out_channels=pre_measured_dim,
                            kernel_size=2,
                            stride=2,
                        ),
                        build_norm_layer(dict(type='BN'), pre_measured_dim)[1],
                        nn.ReLU(True),
                    )
                )

        for i in range(len(in_channels_3d)):
            self.conv_lateral_3d.append(
                nn.Sequential(
                    nn.Conv2d(in_channels_3d[i], self.mink_final_dim, 3, 1, 1, bias=False, device='cuda'),
                    build_norm_layer(dict(type='BN'), self.mink_final_dim)[1],
                    nn.ReLU(True),
                    #     nn.Conv2d(depth_embed_dim, depth_embed_dim, 3, 1, 1, bias=False),
                    #     build_norm_layer(norm_cfg, depth_embed_dim)[1],
                    #     nn.ReLU(True),
                )
            )

            if i != 0:
                self.conv_up_3d.append(
                    nn.Sequential(
                        build_upsample_layer(
                            upsample_cfg,
                            in_channels=self.mink_final_dim,
                            out_channels=self.mink_final_dim,
                            kernel_size=2,
                            stride=2,
                        ),
                        build_norm_layer(dict(type='BN'), self.mink_final_dim)[1],
                        nn.ReLU(True),
                    )
                )

    def forward(self, overshot, premeasured_map, heatmap, data_3d, return_loss=False, **kwargs):
        """
        fp: List[Tensor]
        depth_map: Tensor with shape bs, 1, h, w
        """

        if self.detach_fp is not False and self.detach_fp is not None:
            if isinstance(self.detach_fp, (list, tuple, range)):
                overshot = [it for it in overshot]
                premeasured_map = [it for it in premeasured_map]
                for i in self.detach_fp:
                    overshot[i] = overshot[i].detach()
                    premeasured_map[i] = premeasured_map[i].detach()

            else:
                overshot = [it.detach() for it in overshot]
                premeasured_map = [it.detach() for it in premeasured_map]

        freq_level = kwargs['freq_level']
        # Detour of labels: represents the frequency levels not 3D labels

        # assert not np.isnan(np.array(overshot)).any()
        # assert not torch.isnan(overshot).any()
        # assert not np.isnan(np.array(heatmap)).any()

        mink_output_tmp, final_mesh_coords, final_AP_bounding_box, last_mink_output = self.get_3d_embedding(data_3d)
        # mink_output = mink_output.to_dense()
        # Reshape mink_output

        # print(sinput.shape) # torch.Size([32681, 3])
        # print(mink_output.shape) # torch.Size([32681, 21]): 21 classes

        final_mink_output = []
        last_mink_output = []
        for mink_output in mink_output_tmp:  # mink_output_tmp: (batch #, decoded fea #, ...)
            mink_fea_output = None
            for i in range(len(mink_output)):  # 4 -> RGB
                f = mink_output[len(mink_output) - i - 1]
                if i == 0:
                    last_mink_output.append(f)
                # f = self.conv_lateral_3d[len(mink_output) - i - 1](f)
                # pooling_layer = nn.AdaptiveAvgPool2d((self.interpolate_dim[0], f.size(-1)))
                # f = pooling_layer(f).view(1, 1, self.interpolate_dim[0], f.size(-1))
                interpolate_result = F.interpolate(f, size=self.interpolate_dim, mode='bilinear')

                # conv_lateral is just channel conversion
                # x = torch.cat((f, depth_embed), axis=1)
                # x = f
                # print('current x {}'.format(x.shape))
                if i == 0:
                    mink_fea_output = interpolate_result
                else:
                    # print('current pre_x {}'.format(pre_x.shape)) # in case some odd numbers, nyudepth shape is fixed
                    # adaptive_avg_result = F.adaptive_avg_pool2d(pre_x, output_size=mink_fea_output.shape[-2:])
                    # pooling_layer = nn.AdaptiveAvgPool2d((mink_fea_output.shape[-2:]))
                    # adaptive_avg_result = pooling_layer(pre_x)
                    # assert not torch.isnan(adaptive_avg_result).any(), (adaptive_avg_result, pre_x, mink_fea_output.shape, pre_x.size())
                    mink_fea_output = mink_fea_output + interpolate_result
                    assert not torch.isnan(mink_fea_output).any(), mink_fea_output

                # It is the same size as the ddim random feature map (the same length and width, the number of channels is not necessarily)
                # x is a condition and does not participate in true value regression
            final_mink_output.append(mink_fea_output)

        # pooling again to sync the output size?
        assert len(final_mink_output) == len(final_AP_bounding_box)
        final_AP_bounding_box = torch.stack(final_AP_bounding_box)

        # Embed specific AP's location as bounding box
        # x_min, y_min, z_min, x_max, y_max, z_max
        # fourier_freqs*2*6: fourier_embedder's output shape (2 is sin&cos, 6 is xyzxyz)
        xyzxyz_embedding = self.fourier_embedder(final_AP_bounding_box).cuda()  # B*N*4 --> B*N*C

        # final_mink_output = final_mink_output.view(2, 1, 1, -1)
        # xyzxyz_embedding = xyzxyz_embedding.view(2, 1, -1)
        # Mixture of last_mink_output w/ attention based on the frame index & temporal layer

        final_3d_output = []
        ca_mink_feas = []
        for frame_idx, (mink_output, coord_embedding, mesh_coords, last_mink_feature, freq) in enumerate(
                zip(final_mink_output, xyzxyz_embedding, final_mesh_coords, last_mink_output, freq_level)):

            mink_final = self.mink_embedding_layer(mink_output.reshape(-1)).view(1,
                                                                                 self.mink_final_dim,
                                                                                 self.interpolate_dim[0],
                                                                                 self.interpolate_dim[1])

            # pooling_layer = nn.AdaptiveAvgPool2d((self.mink_final_height, self.mink_final_width))
            # mink_final = pooling_layer(mink_output).view(self.mink_final_dim, self.mink_final_height, self.mink_final_width)
            mink_final = F.interpolate(mink_final, size=(self.mink_final_height, self.mink_final_width),
                                       mode='bilinear') \
                .view(self.mink_final_dim, self.mink_final_height, self.mink_final_width)

            assert not torch.isnan(mink_final).any()
            """
            assert obj_size < self.num_mink_features, str(obj_size) + "/" + str(self.num_mink_features)
            if self.num_mink_features > obj_size:
                padding_size = self.num_mink_features - obj_size
                padding = torch.zeros(padding_size, device='cuda')  # You can use other values instead of zeros
                mink_flatten = torch.cat((mink_flatten, padding), dim=0)
            """
            # mink_final = mink_flatten.view(-1, self.mink_final_height, self.mink_final_width)

            # coord_embedding_min = self.coords_embedding_layer(coord_embedding[0])
            # coord_embedding_max = self.coords_embedding_layer(coord_embedding[1])

            # coord_embedding_min = coord_embedding_min.view(1, self.mink_final_height, self.mink_final_width)
            # coord_embedding_max = coord_embedding_max.view(1, self.mink_final_height, self.mink_final_width)
            # coord_embedding = torch.stack((coord_embedding_max, coord_embedding_min))
            # assert not torch.isnan(coord_embedding_min).any(), coord_embedding_min
            # assert not torch.isnan(coord_embedding_max).any(), coord_embedding_max
            coord_embedding = self.coords_embedding_layer(coord_embedding)
            coord_embedding = coord_embedding.view(1, self.mink_final_height, self.mink_final_width)

            freq_embedding = self.fourier_embedder(freq.view(1, 1)).cuda()  # B*N*4 --> B*N*C
            freq_embedding = self.freq_embedding_layer(freq_embedding)

            freq_embedding = freq_embedding.view(1, self.mink_final_height, self.mink_final_width)

            # print("2: " + str(filtered_coords.size()))
            mesh_embedding = self.fourier_embedder(mesh_coords).cuda()  # B*N*4 --> B*N*C
            mesh_embedding = F.interpolate(
                mesh_embedding.view(1, 1, mesh_embedding.size(0), mesh_embedding.size(1)),
                size=self.interpolate_dim2, mode='bilinear')

            assert not torch.isnan(mesh_embedding).any(), (mesh_embedding, mesh_embedding.size())
            mesh_output = self.mink_embedding_layer2(mesh_embedding.reshape(-1)).view(self.mesh_dim, self.mink_final_height,
                                                                                      self.mink_final_width)
            # print(coords_output)

            # Force the pre-measured signal
            # premeasured_coords = self.get_premeasured_coords(heatmap.size(2), heatmap.size(3))
            # pmsig_embedding = self.get_premeasured_embed(heatmap, premeasured_coords)
            # pmsig_embedding = self.point_embedding_layer(pmsig_embedding)
            # pmsig_embedding = pmsig_embedding.view(1, self.mink_final_height, self.mink_final_width)

            # Cross attention for last mink output
            # last_mink_feature torch.Size([1, 1, 96, 6890])
            seq = last_mink_feature.view(1, last_mink_feature.size(3),
                                         last_mink_feature.size(2)).cuda()  # seq = (1, x, 96)

            queries = torch.rand(self.cross_attention_dim, self.decoder_dim).cuda()
            # queries = self.fourier_embedder(queries).cuda()
            #  queries = queries.view(self.diffusion_inference_steps + 2, -1)  # 20, 16
            # print(queries.size())  torch.Size([20, 16])
            logits = self.perceiver_model_mink(seq, queries=queries)
            logits = logits.view(self.cross_attention_dim, self.mink_final_height, self.mink_final_width)

            concat_input = torch.cat((mink_final, freq_embedding, coord_embedding, mesh_output, logits))

            if frame_idx % self.video_fps == 0:
                mink_fea_output = []

                for mink_output in last_mink_output[frame_idx:frame_idx + self.video_fps]:  # mink_output_tmp: (batch #, decoded fea #, ...)
                    interpolate_result = F.interpolate(mink_output, size=(96, self.video_decoder_dim), mode='bilinear')
                    mink_fea_output.append(interpolate_result)

                video_mink_output = torch.stack(mink_fea_output)  # (fps, coord, feature(96))
                # 1 torch.Size([1, 1, 96, 8357])
                # [10, 6890, 96]
                seq = video_mink_output.view(1, self.video_fps*self.video_decoder_dim, 96)
                queries = torch.zeros(self.video_fps*self.video_cross_attention_dim, self.decoder_dim).cuda()
                # Fill the tensor with consecutive numbers along the first dimension
                for i in range(self.video_fps):
                    queries[i:(i+1)] = i
                # torch.Size([1, 6890, 96]) torch.Size([9, 1200])
                # torch.Size([10, 96, 10000]) torch.Size([10, 10000])
                # torch.Size([1, 6890, 96]) torch.Size([9, 1200]) torch.Size([9, 30, 40])
                # torch.Size([1, 100000, 96]) torch.Size([100000, 1200]) torch.Size([1, 100000, 1200])

                logits = self.perceiver_model_video(seq, queries=queries)

                ca_mink_fea = logits.view(self.video_fps, self.video_cross_attention_dim, self.mink_final_height, self.mink_final_width)
                # final_3d_output.append(ca_mink_fea)
                ca_mink_feas.append(ca_mink_fea)

            final_3d_output.append(concat_input)

        final_3d_output = torch.stack(final_3d_output)
        ca_mink_feas = torch.stack(ca_mink_feas)
        ca_mink_feas = ca_mink_feas.view(-1, ca_mink_feas.size(2), ca_mink_feas.size(3), ca_mink_feas.size(4))

        #   torch.Size([60, 16, 9, 12]) torch.Size([6, 10, 8, 9, 12])
        # depth_map_t = self.depth_transform.t(depth_map)
        # down scale to latent
        heatmap_t = self.depth_transform.t(heatmap)
        # print(heatmap_t.shape) heatmap_t = torch.empty(2, 16, 114, 152, device='cuda') Multi-layer
        # perceptron/artificially set how many channels become depth values latent_depth_mask =
        # nn.functional.adaptive_max_pool2d(depth_mask.float(), output_size=depth_map_t.shape[-2:]) depth =
        # torch.cat((depth_map_t, latent_depth_mask), dim=1)  # bs, 2, h, w if traditional bs, 1+dim, h, w if deep
        image_fea_output = None
        # print(len(overshot))
        # print(overshot[0].size())
        overshot = self.hahineck(overshot)
        premeasured_map = self.hahineck(premeasured_map)

        # overshot = [torch.empty(2, 64, 114, 152, device='cuda'), torch.empty(2, 128, 57, 76, device='cuda'), torch.empty(2, 256, 29, 38, device='cuda')]
        for i in range(len(overshot)):  # 2 -> batch size
            f = overshot[len(overshot) - i - 1]
            image_fea_output = self.conv_lateral[len(overshot) - i - 1](f)

            # conv_lateral is just channel conversion
            # x = torch.cat((f, depth_embed), axis=1)
            # x = f
            # print('current x {}'.format(x.shape))
            if i > 0:
                # print('current pre_x {}'.format(pre_x.shape)) # in case some odd numbers, nyudepth shape is fixed
                # interpolate_result = F.interpolate(f, size=image_fea_output.shape[-2:], mode='bilinear')
                # image_fea_output = image_fea_output + F.adaptive_avg_pool2d(self.conv_up[len(overshot) - i - 1](pre_x), output_size=image_fea_output.shape[-2:])
                image_fea_output = image_fea_output + F.interpolate(self.conv_up[len(overshot) - i - 1](pre_x),
                                                                    size=image_fea_output.shape[-2:], mode='bilinear')

                assert not torch.isnan(image_fea_output).any(), (overshot, image_fea_output)
                # x_model = torch.add(x_model, final_mink_output[i])
            pre_x = image_fea_output
            # It is the same size as the ddim random feature map (the same length and width, the number of channels is not necessarily)
            # x is a condition and does not participate in true value regression
        assert not torch.isnan(image_fea_output).any(), image_fea_output

        image_fea_output2 = None
        for i in range(len(premeasured_map)):  # 2 -> batch size
            f = premeasured_map[len(premeasured_map) - i - 1]
            image_fea_output2 = self.conv_lateral2[len(premeasured_map) - i - 1](f)

            # conv_lateral is just channel conversion
            # x = torch.cat((f, depth_embed), axis=1)
            # x = f
            # print('current x {}'.format(x.shape))
            if i > 0:
                # print('current pre_x {}'.format(pre_x.shape)) # in case some odd numbers, nyudepth shape is fixed
                # interpolate_result = F.interpolate(f, size=image_fea_output.shape[-2:], mode='bilinear')
                # image_fea_output = image_fea_output + F.adaptive_avg_pool2d(self.conv_up[len(overshot) - i - 1](pre_x), output_size=image_fea_output.shape[-2:])
                image_fea_output2 = image_fea_output2 + F.interpolate(
                    self.conv_up2[len(premeasured_map) - i - 1](pre_x),
                    size=image_fea_output2.shape[-2:], mode='bilinear')

                assert not torch.isnan(image_fea_output2).any(), (premeasured_map, image_fea_output2)
                # x_model = torch.add(x_model, final_mink_output[i])
            pre_x = image_fea_output2
            # It is the same size as the ddim random feature map (the same length and width, the number of channels is not necessarily)
            # x is a condition and does not participate in true value regression
        assert not torch.isnan(image_fea_output2).any(), image_fea_output2

        assert image_fea_output.size(0) == final_3d_output.size(0) and image_fea_output.size(2) == final_3d_output.size(
            2) \
               and image_fea_output.size(3) == final_3d_output.size(3), str(image_fea_output.shape) + "/" + str(
            final_3d_output.shape)

        image_fea_output = torch.cat((image_fea_output, image_fea_output2, final_3d_output), dim=1)
        image_fea_output = torch.cat((image_fea_output, ca_mink_feas), dim=1)

        # torch.Size([10, 9, 30, 40]) torch.Size([10, 9, 30, 40]) torch.Size([10, 14, 30, 40])
        # torch.Size([10, 32, 30, 40])
        # torch.Size([10, 1, 30, 40])

        refined_depth_t = self.pipeline(
            batch_size=image_fea_output.shape[0] // self.video_fps,
            video_fps=self.video_fps,
            device=image_fea_output.device,
            dtype=image_fea_output.dtype,
            shape=heatmap_t.shape[-3:],
            # shape=x.shape[-3:],
            input_args=(
                image_fea_output,
                None,
                None,
                None
            ),
            num_inference_steps=self.diffusion_inference_steps,
            return_dict=False,
        )
        # print('final_latent_output {}'.format(refined_depth_t.shape))
        refined_depth = self.depth_transform.inv_t(refined_depth_t)
        heatmap = self.depth_transform.inv_t(heatmap_t)

        # The refine depth is output directly, there is no cspn module yet
        diffusion_loss = self.diffusion_loss(
            pred_depth=refined_depth,
            gt_depth=heatmap_t,
            video_fps=self.video_fps,
            refine_module_inputs=(
                image_fea_output,
                None,
                None,
                None
            ),
            blur_depth_t=refined_depth_t,
            weight=1.0)

        msig_loss = self.msig_loss(
            pred_depth=refined_depth,
            gt_depth=heatmap,
            premeasured_coords=kwargs['premeasured_coords'])

        assert not torch.isnan(diffusion_loss).any(), (refined_depth, heatmap_t, image_fea_output, diffusion_loss)
        mink_output = {'pred': refined_depth, 'pred_init': heatmap, 'blur_depth_t': heatmap_t,
                       'ddim_loss': diffusion_loss, 'msig_loss': msig_loss, 'gt_map_t': heatmap_t,
                       'pred_uncertainty': None,
                       'pred_inter': None, 'weight_map': None,
                       'guidance': None, 'offset': None, 'aff': None,
                       'gamma': None, 'confidence': None}

        return mink_output

    def loss(self, pred_depth, gt_depth, refine_module_inputs, blur_depth_t, pred_uncertainty=None, weight_map=None,
             **kwargs):
        loss_dict = super().loss(pred_depth, gt_depth, pred_uncertainty, weight_map, **kwargs)
        for loss_cfg in self.loss_cfgs:
            loss_fnc_name = loss_cfg['loss_func']
            loss_key = loss_cfg['name']
            if loss_key == 'ddim_loss':
                loss_fnc = self.diffusion_loss
            else:
                continue
            loss = loss_fnc(
                pred_depth=pred_depth, pred_uncertainty=pred_uncertainty,
                gt_depth=gt_depth,
                refine_module_inputs=refine_module_inputs,
                blur_depth_t=blur_depth_t,
                weight_map=weight_map, **loss_cfg, **kwargs
            )
            loss_dict[loss_key] = loss
        return loss_dict

    def diffusion_loss(self, gt_depth, video_fps, refine_module_inputs, blur_depth_t, weight, **kwargs):
        # Sample noise to add to the images
        noise = torch.randn(blur_depth_t.shape).to(blur_depth_t.device)
        bs = int(blur_depth_t.shape[0] // video_fps)
        # TODO why one batch in here?
        # bs = blur_depth_t.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.scheduler.num_train_timesteps, (bs,), device=gt_depth.device).long()
        # The randomness here is in the bs dimension, which cannot be too small.
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        timesteps_repeated = timesteps.repeat(video_fps)

        noisy_images = self.scheduler.add_noise(blur_depth_t, noise, timesteps_repeated)

        noise_pred = self.model(noisy_images, timesteps, *refine_module_inputs)

        loss = F.mse_loss(noise_pred, noise)

        return loss

    def msig_loss(self, pred_depth, gt_depth, premeasured_coords):
        # Sample noise to add to the images
        assert pred_depth.shape == gt_depth.shape, print(pred_depth.shape, gt_depth.shape)
        pred_result = []
        gt_result = []
        total_loss = 0
        for batch_idx in range(len(premeasured_coords)):
            tmp_pred_result = torch.tensor([pred_depth[batch_idx][0][i][j] for i, j in premeasured_coords[batch_idx]]).cuda()
            tmp_gt_result = torch.tensor([gt_depth[batch_idx][0][i][j] for i, j in premeasured_coords[batch_idx]]).cuda()
            pred_result.append(tmp_pred_result)
            gt_result.append(tmp_gt_result)
            # loss = F.mse_loss(tmp_pred_result, tmp_gt_result)

            # total_loss += loss
        pred_result = torch.stack(pred_result, dim=0)
        gt_result = torch.stack(gt_result, dim=0)

        loss = F.mse_loss(pred_result, gt_result)

        return loss

    def ddim_loss_gt(self, gt_depth, refine_module_inputs, blur_depth_t, weight, **kwargs):
        # Sample noise to add to the images
        noise = torch.randn(gt_depth.shape).to(gt_depth.device)
        bs = gt_depth.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.scheduler.num_train_timesteps, (bs,), device=gt_depth.device).long()
        # 这里的随机是在 bs维度，这个情况不能太小。
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_images = self.scheduler.add_noise(gt_depth, noise, timesteps)

        noise_pred = self.model(noisy_images, timesteps, *refine_module_inputs)

        loss = F.mse_loss(noise_pred, noise)

        return loss

    def get_premeasured_embed(self, heatmap, pre_coords):
        selected_values = torch.tensor([heatmap[0, 0, i, j] for i, j in pre_coords])
        pre_coords = pre_coords.astype('float32')
        pre_coords[:, 0] /= np.array(heatmap.size(2), dtype=np.float32)
        pre_coords[:, 1] /= np.array(heatmap.size(3), dtype=np.float32)
        pre_coords = torch.from_numpy(pre_coords)
        assert torch.all(pre_coords <= 1), pre_coords

        pre_coords = pre_coords.view(-1).cuda()
        selected_values = selected_values.view(-1).cuda()
        input_data = torch.cat((pre_coords, selected_values), dim=0)
        return input_data

    def get_3d_embedding(self, data_3d):
        (coords, feats, _) = data_3d
        # Remove * 100 because our problem is different from origianl classification problems
        # Our problem has fixed env.
        # TODO right ???
        coords[:, 1:] += (torch.rand(3) * 100).type_as(coords)

        tmp_coords = []
        total_coords = []
        tmp_feats = []
        cur_batch_idx = torch.tensor(1, device='cuda', dtype=torch.int32)
        mink_output_tmp = []
        tmp_AP_bounding_box = []
        final_AP_bounding_box = []
        tmp_mesh_coords = []
        final_mesh_coords = []
        last_mink_list = []
        assert coords.size(0) == feats.size(0)

        ap_tmp = torch.tensor([255., 255., 255.], device='cuda')
        box_tmp = torch.tensor([0., 0., 255.], device='cuda')
        for coord, feat in zip(coords, feats):
            batch_idx = coord.data[0]

            if cur_batch_idx.item() == batch_idx:
                tmp_coords.append(coord)
                tmp_feats.append(feat)

                # AP location
                if torch.equal(feat, ap_tmp):
                    tmp_AP_bounding_box.append(coord[1:])
                # elif feat == [0., 0., 255.]:
                elif torch.equal(feat, box_tmp):
                    tmp_mesh_coords.append(coord[1:])

            else:
                tmp_coords = torch.stack(tmp_coords)
                total_coords.append(tmp_coords)
                tmp_feats = torch.stack(tmp_feats)

                # tmp_feats = tmp_feats.to(dtype=torch.float16)
                # tmp_coords = tmp_coords.to(dtype=torch.float16)

                sinput = SparseTensor(tmp_feats.cuda(non_blocking=True), tmp_coords.cuda(non_blocking=True),
                                      device='cuda')
                mink_output, last_mink_output = self.mink_model(sinput)
                last_mink_list.append(last_mink_output)
                mink_output_tmp.append(mink_output)

                # AP
                assert tmp_AP_bounding_box
                tmp_AP_bounding_box = torch.stack(tmp_AP_bounding_box)
                min_coords, _ = torch.min(tmp_AP_bounding_box, 0)
                max_coords, _ = torch.max(tmp_AP_bounding_box, 0)

                # Mesh
                assert tmp_mesh_coords
                tmp_mesh_coords = torch.stack(tmp_mesh_coords)
                final_mesh_coords.append(tmp_mesh_coords)

                final_AP_bounding_box.append(torch.cat((min_coords, max_coords)))

                tmp_coords = []
                tmp_feats = []
                tmp_mesh_coords = []
                tmp_AP_bounding_box = []
                cur_batch_idx = batch_idx

        # Last batch
        tmp_coords = torch.stack(tmp_coords)
        total_coords.append(tmp_coords)
        tmp_feats = torch.stack(tmp_feats)
        sinput = SparseTensor(tmp_feats.cuda(non_blocking=True), tmp_coords.cuda(non_blocking=True), device='cuda')

        mink_output, last_mink_output = self.mink_model(sinput)
        last_mink_list.append(last_mink_output)
        mink_output_tmp.append(mink_output)

        if not tmp_AP_bounding_box:
            print("unexpected")
            min_coords = torch.tensor([0, 0, 0])
            max_coords = torch.tensor([0, 0, 0])
        else:
            tmp_AP_bounding_box = torch.stack(tmp_AP_bounding_box)
            min_coords, _ = torch.min(tmp_AP_bounding_box, 0)
            max_coords, _ = torch.max(tmp_AP_bounding_box, 0)
        # final_AP_bounding_box.append(torch.stack((min_coords, max_coords), dim=0))
        final_AP_bounding_box.append(torch.cat((min_coords, max_coords)))

        # Mesh
        if not tmp_mesh_coords:
            print("unexpected")
            final_mesh_coords.append(torch.tensor([[0, 0, 0]]))
        else:
            tmp_mesh_coords = torch.stack(tmp_mesh_coords)
            final_mesh_coords.append(tmp_mesh_coords)
        return mink_output_tmp, final_mesh_coords, final_AP_bounding_box, last_mink_list


class CNNDDIMPipiline:
    '''
    Modified from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/ddim/pipeline_ddim.py
    '''

    def __init__(self, model, scheduler):
        super().__init__()
        self.model = model
        self.scheduler = scheduler

    def __call__(
            self,
            batch_size,
            video_fps,
            device,
            dtype,
            shape,
            input_args,
            generator: Optional[torch.Generator] = None,
            eta: float = 0.0,
            num_inference_steps: int = 50,
            return_dict: bool = True,
            **kwargs,
    ) -> Union[Dict, Tuple]:
        if generator is not None and generator.device.type != self.device.type and self.device.type != "mps":
            message = (
                f"The `generator` device is `{generator.device}` and does not match the pipeline "
                f"device `{self.device}`, so the `generator` will be ignored. "
                f'Please use `generator=torch.Generator(device="{self.device}")` instead.'
            )
            raise RuntimeError(
                "generator.device == 'cpu'",
                "0.11.0",
                message,
            )
            generator = None

        # Sample gaussian noise to begin loop
        #  image_shape (1, 16, 176, 353)

        # Video diffusion setting -> batch size * input sequence (fps)
        image_shape = (batch_size * video_fps, *shape)
        image = torch.randn(image_shape, generator=generator, device=device, dtype=dtype)
        # print('random_noise is {}'.format(image.shape))
        # set step values
        self.scheduler.set_timesteps(num_inference_steps)
        for t in self.scheduler.timesteps:
            # timesteps selected 20 steps
            # 1. predict noise model_output
            # self.model: ScheduledCNNRefine
            model_output = self.model(image, t.to(device), *input_args)
            # model_output torch.Size([1, 16, 176, 353])
            # 2. predict previous mean of image x_t-1 and add variance depending on eta
            # eta corresponds to η in paper and should be between [0, 1]
            # do x_t -> x_t-1
            image = self.scheduler.step(
                model_output, t, image, eta=eta, use_clipped_model_output=True, generator=generator
            )['prev_sample']
            # image torch.Size([1, 16, 176, 353])

        if not return_dict:
            return image

        return {'images': image}


class ScheduledCNNRefine(BaseModule):
    def __init__(self, channels_in, channels_noise, video_fps, **kwargs):
        super().__init__(**kwargs)
        self.do_classifier_free_guidance = None
        mid_channel = channels_in
        group_num = 4
        num_attention_heads = 8
        transformer_layers_per_block = 1
        self.video_fps = video_fps
        self.cross_attention_dim: int = 640
        block_out_channels = (320, 640, 1280, 1280)
        self.time_channel = self.video_fps
        self.motion_bucket_id = 127
        self.noise_aug_strength = 0.02
        self.num_videos_per_prompt = 1
        self.addition_time_embed_dim: int = 256
        self.projection_class_embeddings_input_dim: int = 1536
        self.time_embed_dim = self.time_channel * 4

        # channels_noise = 16
        # channels_in = refined_channels_in
        self.noise_embedding = nn.Sequential(
            nn.Conv2d(channels_noise, 64, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(4, 64),
            nn.ReLU(True),
            nn.Conv2d(64, channels_in, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(4, channels_in),
            nn.ReLU(True),
        )

        self.upsample_fuse = UpSample_add(channels_in, channels_in)

        self.time_embedding = nn.Embedding(1280, channels_in)
        self.time_video_embedding = TimestepEmbedding(self.time_channel, self.time_embed_dim)
        self.time_proj = Timesteps(self.time_channel, True, downscale_freq_shift=0)

        self.pred = nn.Sequential(
            nn.Conv2d(channels_in, 64, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(4, 64),
            nn.ReLU(True),
            nn.Conv2d(64, channels_noise, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(4, channels_noise),
            nn.ReLU(True),
        )
        self.spatio_temp_block = SpatioTemporalResBlock(
            in_channels=mid_channel,
            out_channels=64,
            temb_channels=self.video_fps * 4,
            groups=group_num,
            eps=1e-6,
            temporal_eps=1e-5,
        )
        self.spatio_temp_block2 = SpatioTemporalResBlock(
            in_channels=64,
            out_channels=mid_channel,
            temb_channels=self.video_fps * 4,
            groups=group_num,
            eps=1e-6,
            temporal_eps=1e-5,
        )
        assert mid_channel % num_attention_heads == 0, (mid_channel, num_attention_heads)

        self.transformer_spatio = TransformerSpatioTemporalModel(
            num_attention_heads,
            64 // num_attention_heads,
            in_channels=64,
            num_layers=transformer_layers_per_block,
            # cross_attention_dim=self.cross_attention_dim,
            cross_attention_dim=None,
            groups=group_num,
        )
        self.transformer_spatio2 = TransformerSpatioTemporalModel(
            num_attention_heads,
            mid_channel // num_attention_heads,
            in_channels=mid_channel,
            num_layers=transformer_layers_per_block,
            # cross_attention_dim=self.cross_attention_dim,
            cross_attention_dim=None,
            groups=group_num,
        )

    def forward(self, noisy_image, t, *args):
        feat, blur_depth, sparse_depth, sparse_mask = args
        # print('debug: feat shape {}'.format(feat.shape))
        # diff = (noisy_image - blur_depth).abs()
        timesteps = t
        num_frames = self.video_fps
        self.batch_size = noisy_image.shape[0] // num_frames

        if t.numel() == self.batch_size:
            t = t.repeat_interleave(num_frames)
        # print("before t:", t)
        if t.numel() == 1:
            # print(t)
            feat = feat + self.time_embedding(t)[..., None, None]
            # feat = feat + self.time_embedding(t)[None, :, None, None]
            # If t itself is a value, the first bs dimension needs to be expanded (this is temporarily not applicable)
        else:
            # print(t)
            feat = feat + self.time_embedding(t)[..., None, None]
        # layer(feat) - noise_image
        # blur_depth = self.layer(feat);
        # ret =  a* noisy_image - b * blur_depth
        # print('debug: noisy_image shape {}'.format(noisy_image.shape))
        # feat = feat + self.noise_embedding(noisy_image)

        # feat, (Batch<1>, Frame<10>, C, H, W) -> (Batch<10>, C, H, W)
        # 5 dim -> 4 dim by batch*time
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            if isinstance(timesteps, float):
                dtype = torch.float64
            else:
                dtype = torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device="cuda")
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to("cuda")

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        if len(timesteps) == 1:
            timesteps = timesteps.expand(self.batch_size)
        # timesteps = timesteps.unsqueeze(0)

        t_emb = self.time_proj(timesteps)

        # `Timesteps` does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(torch.float32)

        emb = self.time_video_embedding(t_emb)

        self.do_classifier_free_guidance = True
        added_time_ids = self._get_add_time_ids(
            self.video_fps - 1,
            self.motion_bucket_id,
            self.noise_aug_strength,
            torch.float32,
            self.batch_size,
            self.num_videos_per_prompt,
            self.do_classifier_free_guidance,
        )
        added_time_ids = added_time_ids.to("cuda")

        self.add_time_proj = Timesteps(self.addition_time_embed_dim, True, downscale_freq_shift=0).cuda()
        self.add_embedding = TimestepEmbedding(self.projection_class_embeddings_input_dim, self.time_embed_dim).cuda()
        # time_embed_dim = block_out_channels[0] * 4

        time_embeds = self.add_time_proj(added_time_ids.flatten())
        # print(time_embeds.size())
        time_embeds = time_embeds.reshape((self.batch_size, -1)).to("cuda")
        aug_emb = self.add_embedding(time_embeds)
        emb = emb + aug_emb

        # Flatten the batch and frames dimensions
        # sample: [batch, frames, channels, height, width] -> [batch * frames, channels, height, width]
        # Repeat the embeddings num_video_frames times
        # emb: [batch, channels] -> [batch * frames, channels]
        emb = emb.repeat_interleave(num_frames, dim=0)

        # if emb is not None:
        # emb = emb.reshape(1, num_frames, -1)

        # print(noisy_image.size())  # torch.Size([30, 16, 176, 353])
        image_only_indicator = torch.zeros(self.batch_size, num_frames, dtype=torch.long, device="cuda")
        encoder_hidden_states = feat
        encoder_hidden_states = encoder_hidden_states.reshape(num_frames*self.batch_size, 1, -1)

        noisy_hidden_states = self.noise_embedding(noisy_image)  # B = FPS, C, H, W

        noisy_hidden_states = self.spatio_temp_block(noisy_hidden_states, temb=emb, image_only_indicator=image_only_indicator)
        noisy_hidden_states = self.transformer_spatio(noisy_hidden_states, image_only_indicator=image_only_indicator, encoder_hidden_states=encoder_hidden_states)[0]
        noisy_hidden_states = self.spatio_temp_block2(noisy_hidden_states, temb=emb, image_only_indicator=image_only_indicator)
        noisy_hidden_states = self.transformer_spatio2(noisy_hidden_states, image_only_indicator=image_only_indicator, encoder_hidden_states=encoder_hidden_states)[0]

        feat_hidden_states = self.spatio_temp_block(feat, temb=emb, image_only_indicator=image_only_indicator)
        feat_hidden_states = self.transformer_spatio(feat_hidden_states, image_only_indicator=image_only_indicator, encoder_hidden_states=encoder_hidden_states)[0]
        feat_hidden_states = self.spatio_temp_block2(feat_hidden_states, temb=emb, image_only_indicator=image_only_indicator)
        feat_hidden_states = self.transformer_spatio2(feat_hidden_states, image_only_indicator=image_only_indicator, encoder_hidden_states=encoder_hidden_states)[0]

        feat = self.upsample_fuse(feat_hidden_states, noisy_hidden_states)

        ret = self.pred(feat)

        return ret

    def _get_add_time_ids(
            self,
            fps,
            motion_bucket_id,
            noise_aug_strength,
            dtype,
            batch_size,
            num_videos_per_prompt,
            do_classifier_free_guidance,
    ):
        add_time_ids = [fps, motion_bucket_id, noise_aug_strength]
        """
        
        passed_add_embed_dim = addition_time_embed_dim * len(add_time_ids)
        # expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features
        expected_add_embed_dim = 320
        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )
        """
        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        add_time_ids = add_time_ids.repeat(batch_size * num_videos_per_prompt, 1)

        if do_classifier_free_guidance:
            add_time_ids = torch.cat([add_time_ids, add_time_ids])

        return add_time_ids


class UpSample(nn.Sequential):
    '''Fusion module
    From Adabins

    '''

    def __init__(self, skip_input, output_features, conv_cfg=None, norm_cfg=None, act_cfg=None):
        super(UpSample, self).__init__()
        self.convA = ConvModule(skip_input, output_features, kernel_size=3, stride=1, padding=1, conv_cfg=conv_cfg,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.convB = ConvModule(output_features, output_features, kernel_size=3, stride=1, padding=1, conv_cfg=conv_cfg,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, x, concat_with):
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)
        return self.convB(self.convA(torch.cat([up_x, concat_with], dim=1)))


class UpSample_add(nn.Sequential):
    '''Fusion module
    From Adabins

    '''

    def __init__(self, skip_input, output_features, conv_cfg=None, norm_cfg=None, act_cfg=None):
        super(UpSample_add, self).__init__()
        self.convA = ConvModule(skip_input, output_features, kernel_size=3, stride=1, padding=1, conv_cfg=conv_cfg,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.convB = ConvModule(output_features, output_features, kernel_size=3, stride=1, padding=1, conv_cfg=conv_cfg,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, x, concat_with):
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)
        return self.convB(self.convA(up_x + concat_with))
