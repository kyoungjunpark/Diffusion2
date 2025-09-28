# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn
import torch.nn.functional as F
from mmdet3d.models.builder import HEADS, build_loss
from mmcv.runner import BaseModule, ModuleList, force_fp32
from mmcv.cnn import ConvModule, build_conv_layer, build_norm_layer, build_upsample_layer
from model.diffusers.schedulers.scheduling_ddim import DDIMScheduler
from model.diffusers.schedulers.scheduling_ddpm import DDPMScheduler

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

from perceiver_pytorch import PerceiverIO

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


@HEADS.register_module()
class Image_Diffusion(BaseDepthRefine):
    def __init__(
            self,
            up_scale_factor=1,
            inference_steps=20,
            num_train_timesteps=1000,
            return_indices=None,
            mink_pretrained_path=None,
            diff_type=None,
            image_width=None,
            image_height=None,
            depth_transform_cfg=dict(type='DeepDepthTransformWithUpsamplingHeatmap_1D', hidden=16, eps=1e-6),
            ap_loc=True,
            cross_attention=True,
            **kwargs
    ):
        super().__init__(blur_depth_head=False, **kwargs)
        # channels_in = kwargs['in_channels'][0] + self.depth_embed_dim
        self.mink_final_width = int(round(image_width / 4))
        self.mink_final_height = int(round(image_height / 4))
        self.ap_loc = ap_loc
        self.cross_attention = cross_attention

        # Pre measured number of points
        self.cross_attention_dim = 64

        if self.ap_loc:
            fpn_dim = 126
            fpn_dim2 = 209
            self.mink_final_dim = 1
            refined_channels_in = fpn_dim + fpn_dim2 + self.mink_final_dim + 4
        else:
            fpn_dim = 256
            fpn_dim2 = 256
            self.mink_final_dim = 4
            refined_channels_in = fpn_dim + fpn_dim2 + self.mink_final_dim

        if self.cross_attention:
            fpn_dim -= self.cross_attention_dim

        in_channels = [192, 384, 768, 1536]
        # in_channels_3d = [96, 96, 128, 128]
        in_channels_3d = [1, 1, 1]

        self.interpolate_dim = (64, 30000)
        # self.interpolate_dim = (64, 20000)

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
            nn.Linear(mink2_inner_features, 2 * self.mink_final_height * self.mink_final_width, device='cuda'),
        )
        self.depth_transform = DEPTH_TRANSFORM.build(depth_transform_cfg)
        self.return_indices = return_indices
        self.model = ScheduledCNNRefine(channels_in=refined_channels_in, channels_noise=kwargs['depth_feature_dim'], )
        self.diffusion_inference_steps = inference_steps
        if diff_type == 'DDIM':
            self.scheduler = DDIMScheduler(num_train_timesteps=num_train_timesteps, clip_sample=False)
            self.pipeline = CNNDDIMPipiline(self.model, self.scheduler)
        elif diff_type == 'DDPM':
            self.scheduler = DDPMScheduler(num_train_timesteps=num_train_timesteps, clip_sample=False)
            self.pipeline = CNNDDPMPipiline(self.model, self.scheduler)
        del self.weight_head
        del self.conv_lateral
        del self.conv_up

        upsample_cfg = dict(type='deconv', bias=False)
        self.hahineck = HAHIHeteroNeck(in_channels=in_channels, out_channels=in_channels,
                                       embedding_dim=512,
                                       positional_encoding=dict(type='SinePositionalEncoding', num_feats=256),
                                       scales=[1, 1, 1, 1], cross_att=False, self_att=False, num_points=8).cuda()
        d_model = d_embed = 512
        h = 8
        d_ff = 2048
        dr_rate = 0.1
        n_layer = 6
        norm_eps = 1e-5
        query_batch_num = 10
        """
        import copy
        copy = copy.deepcopy
        norm = nn.LayerNorm(d_embed, eps=norm_eps)
        attention = MultiHeadAttentionLayer(
            d_model=d_model,
            h=h,
            qkv_fc=nn.Linear(d_embed, d_model),
            out_fc=nn.Linear(d_model, d_embed),
            dr_rate=dr_rate)
        position_ff = PositionWiseFeedForwardLayer(
            fc1=nn.Linear(d_embed, d_ff),
            fc2=nn.Linear(d_ff, d_embed),
            dr_rate=dr_rate)
        decoder_block = DecoderBlock(
            self_attention=copy(attention),
            cross_attention=copy(attention),
            position_ff=copy(position_ff),
            norm=copy(norm),
            dr_rate=dr_rate)
        decoder = Decoder(
            decoder_block=decoder_block,
            n_layer=n_layer,
            norm=copy(norm))
        self.multihead_attn = nn.MultiheadAttention(d_embed, h)
        """
        self.decoder_dim = self.mink_final_width * self.mink_final_height
        self.perceiver_model = PerceiverIO(
            dim=96,  # dimension of sequence to be encoded
            queries_dim=self.decoder_dim,  # dimension of decoder queries
            # logits_dim=708,  # 177*88/22=708, +2 for just fitting into 177 x 88
            logits_dim=None,
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
        # self.queries = torch.rand(64, decoder_dim).cuda()
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
                    nn.Conv2d(in_channels[i], fpn_dim, 3, 1, 1, bias=False, device='cuda'),
                    build_norm_layer(dict(type='BN'), fpn_dim)[1],
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
                            in_channels=fpn_dim,
                            out_channels=fpn_dim,
                            kernel_size=2,
                            stride=2,
                        ),
                        build_norm_layer(dict(type='BN'), fpn_dim)[1],
                        nn.ReLU(True),
                    )
                )

        for i in range(len(in_channels)):
            self.conv_lateral2.append(
                nn.Sequential(
                    nn.Conv2d(in_channels[i], fpn_dim2, 3, 1, 1, bias=False, device='cuda'),
                    build_norm_layer(dict(type='BN'), fpn_dim2)[1],
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
                            in_channels=fpn_dim2,
                            out_channels=fpn_dim2,
                            kernel_size=2,
                            stride=2,
                        ),
                        build_norm_layer(dict(type='BN'), fpn_dim2)[1],
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

        final_3d_output = []
        for batch_idx, (mink_output, coord_embedding, mesh_coords, last_mink_feature, freq) in enumerate(
                zip(final_mink_output, xyzxyz_embedding, final_mesh_coords, last_mink_output, freq_level)):

            mink_output = self.mink_embedding_layer(mink_output.reshape(-1)).view(1,
                                                                                  self.mink_final_dim,
                                                                                  self.interpolate_dim[0],
                                                                                  self.interpolate_dim[1])

            pooling_layer = nn.AdaptiveAvgPool2d((self.mink_final_height, self.mink_final_width))
            # mink_final = pooling_layer(mink_output).view(self.mink_final_dim, self.mink_final_height, self.mink_final_width)
            mink_final = F.interpolate(mink_output, size=(self.mink_final_height, self.mink_final_width),
                                       mode='bilinear') \
                .view(self.mink_final_dim, self.mink_final_height, self.mink_final_width)

            assert not torch.isnan(mink_final).any(), (
                mink_output, mink_output_tmp, coords, feats, mink_output.size(), coords.size())

            """
            assert obj_size < self.num_mink_features, str(obj_size) + "/" + str(self.num_mink_features)
            if self.num_mink_features > obj_size:
                padding_size = self.num_mink_features - obj_size
                padding = torch.zeros(padding_size, device='cuda')  # You can use other values instead of zeros
                mink_flatten = torch.cat((mink_flatten, padding), dim=0)
            """
            # mink_final = mink_flatten.view(-1, self.mink_final_height, self.mink_final_width)

            if self.ap_loc:
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
                mesh_output = self.mink_embedding_layer2(mesh_embedding.reshape(-1)).view(2, self.mink_final_height,
                                                                                          self.mink_final_width)
                # print(coords_output)

                # Force the pre-measured signal
                # premeasured_coords = self.get_premeasured_coords(heatmap.size(2), heatmap.size(3))
                # pmsig_embedding = self.get_premeasured_embed(heatmap, premeasured_coords)
                # pmsig_embedding = self.point_embedding_layer(pmsig_embedding)
                # pmsig_embedding = pmsig_embedding.view(1, self.mink_final_height, self.mink_final_width)

                if self.cross_attention:
                    # Cross attention for last mink output
                    seq = last_mink_feature.view(1, last_mink_feature.size(3),
                                                 last_mink_feature.size(2)).cuda()  # seq = (1, x, 96)
                    # queries = torch.randn(self.diffusion_inference_steps, 96).cuda()
                    # queries = torch.tensor(list(range(1, 21 + 2))).cuda()  # 2 for just fit into 177x88
                    # queries = torch.tensor((range(1, 64))).cuda()  # 2 for just fit into 177x88
                    # queries = torch.rand(64).cuda()
                    queries = torch.rand(self.cross_attention_dim, self.decoder_dim).cuda()
                    # TODO: Adding time_embedding if possible

                    # queries = self.fourier_embedder(queries).cuda()
                    #  queries = queries.view(self.diffusion_inference_steps + 2, -1)  # 20, 16

                    # print(queries.size())  torch.Size([20, 16])
                    logits = self.perceiver_model(seq, queries=queries)
                    # print(seq.size()) torch.Size([1, 11082, 96])
                    # print(logits.size()) # torch.Size([1, 20, 20])
                    # torch.Size([1, 128, 20])
                    logits = logits.view(self.cross_attention_dim, self.mink_final_height, self.mink_final_width)
                    concat_input = torch.cat((mink_final, freq_embedding, coord_embedding, mesh_output, logits))
                else:
                    concat_input = torch.cat((mink_final, freq_embedding, coord_embedding, mesh_output))

                # concat_input = torch.cat((mink_final, coord_embedding, mesh_output))

                # concat_input = torch.cat((mink_final, coord_embedding_min, coord_embedding_max, mesh_output))
                final_3d_output.append(concat_input)
            else:
                final_3d_output.append(mink_final)
        final_3d_output = torch.stack(final_3d_output)
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
        # here3:  4 torch.Size([2, 192, 80, 160])
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
                image_fea_output2 = image_fea_output2 + F.interpolate(self.conv_up2[len(premeasured_map) - i - 1](pre_x),
                                                                    size=image_fea_output2.shape[-2:], mode='bilinear')

                assert not torch.isnan(image_fea_output2).any(), (premeasured_map, image_fea_output2)
                # x_model = torch.add(x_model, final_mink_output[i])
            pre_x = image_fea_output2
            # It is the same size as the ddim random feature map (the same length and width, the number of channels is not necessarily)
            # x is a condition and does not participate in true value regression
        assert not torch.isnan(image_fea_output2).any(), image_fea_output2
        # x = self.convup_fp(x)
        # print(x_model.shape)
        # x_model = torch.empty(2,reshape_layer 256, 114, 152, device='cuda')
        # x_model = self._2d(x_model.flatten()).view(128, self.mink_final_widthz)

        # upscale x into depth real size will crush the me
        # 3D features from Minknet
        # size_a = mink_output.size() # 2d
        # size_b = x_model.size() # 4d

        # Assuming that the last dimension of output_a doesn't match the last dimension of output_b
        # We will add a fully connected layer to output_a to adjust its dimensions
        # assert size_a[-1] != sizfrefined_depthe_b[-1]
        # Define the fully connected layer with the desired output dimension
        # fc_layer = nn.Linear(in_features=size_a[-1], out_features=size_b[-1], device='cuda')

        # Apply the fully connected layer to output_a
        # adjusted_output_a = fc_layer(mink_output)

        # Concatenate the adjusted outputs
        # concatenated_output = torch.cat((adjusted_output_a, mink_output), dim=1)
        # x_model += adjusted_output_a
        # print(x_model.shape)  # torch.Size([3, 256, 114, 152]) batch, channel(RGB), x, y
        # print(final_mink_output.shape)

        # torch.Size([3, 32681, 21]) flatten into 2d(3, -1) -> mutify
        # -> reshape into the x_model(undown liner) ->
        #
        # 2d feature: [3, 256, 114, 152]
        # 3d feature: [3, 2, 114, 152]
        # -> my feature: [3, 258, 114, 152]
        assert image_fea_output.size(0) == final_3d_output.size(0) and image_fea_output.size(2) == final_3d_output.size(
            2) \
               and image_fea_output.size(3) == final_3d_output.size(3), str(image_fea_output.shape) + "/" + str(
            final_3d_output.shape)

        image_fea_output = torch.cat((image_fea_output, image_fea_output2, final_3d_output), dim=1)

        # reshape mink -> x_model (maintain w, h) (x channel) -> (2d -> 4d tensor) undown liner-> same shape
        #
        # Add new linear layout (w/ relu)  -> x_model shape
        # print(image_fea_output.size()) torch.Size([1, 260, 88, 177])
        # heatmap_t shape:  torch.Size([16, 176, 353])
        refined_depth_t = self.pipeline(
            batch_size=image_fea_output.shape[0],
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
        # heatmap = self.depth_transform.inv_t(heatmap_t)

        # The refine depth is output directly, there is no cspn module yet
        diffusion_loss = self.diffusion_loss(
            pred_depth=refined_depth,
            gt_depth=heatmap_t,
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

    def diffusion_loss(self, gt_depth, refine_module_inputs, blur_depth_t, weight, **kwargs):
        # Sample noise to add to the images
        noise = torch.randn(blur_depth_t.shape).to(blur_depth_t.device)
        bs = blur_depth_t.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.scheduler.num_train_timesteps, (bs,), device=gt_depth.device).long()
        # The randomness here is in the bs dimension, which cannot be too small.
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_images = self.scheduler.add_noise(blur_depth_t, noise, timesteps)

        noise_pred = self.model(noisy_images, timesteps, *refine_module_inputs)

        loss = F.mse_loss(noise_pred, noise)

        return loss

    def msig_loss(self, pred_depth, gt_depth, premeasured_coords):
        # Sample noise to add to the images
        assert pred_depth.shape == gt_depth.shape, (pred_depth.shape, gt_depth.shape)
        pred_result = []
        gt_result = []
        for batch_idx in range(len(premeasured_coords)):
            tmp_pred_result = torch.tensor([pred_depth[batch_idx][0][i][j] for i, j in premeasured_coords[batch_idx]]).cuda()
            tmp_gt_result = torch.tensor([gt_depth[batch_idx][0][i][j] for i, j in premeasured_coords[batch_idx]]).cuda()
            pred_result.append(tmp_pred_result)
            gt_result.append(tmp_gt_result)

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
        cur_batch_idx = torch.tensor(0, device='cuda', dtype=torch.int32)
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
        image_shape = (batch_size, *shape)

        image = torch.randn(image_shape, generator=generator, device=device, dtype=dtype)

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


class CNNDDPMPipiline:
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
            device,
            dtype,
            shape,
            input_args,
            generator: Optional[torch.Generator] = None,
            eta: float = 0.0,
            num_inference_steps: int = 1000,
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
        image_shape = (batch_size, *shape)

        image = torch.randn(image_shape, generator=generator, device=device, dtype=dtype)
        # print('random_noise is {}'.format(image.shape))
        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        # Changing input_args?
        for t in self.scheduler.timesteps:
            # timesteps selected 20 steps
            # 1. predict noise model_output
            model_output = self.model(image, t.to(device), *input_args)

            # 2. compute previous image: x_t -> x_t-1
            image = self.scheduler.step(model_output, t, image, generator=generator)['prev_sample']

        if not return_dict:
            return image
        return {'images': image}


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


class ScheduledCNNRefine(BaseModule):
    def __init__(self, channels_in, channels_noise, **kwargs):
        super().__init__(**kwargs)
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

        self.pred = nn.Sequential(
            nn.Conv2d(channels_in, 64, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(4, 64),
            nn.ReLU(True),
            nn.Conv2d(64, channels_noise, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(4, channels_noise),
            nn.ReLU(True),
        )

    def forward(self, noisy_image, t, *args):
        feat, blur_depth, sparse_depth, sparse_mask = args
        # print('debug: feat shape {}'.format(feat.shape))
        # diff = (noisy_image - blur_depth).abs()
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
        feat = self.upsample_fuse(feat, self.noise_embedding(noisy_image))

        ret = self.pred(feat)

        return ret

