import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.unet.decoder import DecoderBlock
from segmentation_models_pytorch.base import SegmentationHead
from segmentation_models_pytorch.base import initialization as init


class UnetDecoder(nn.Module):

    def __init__(self, encoder_channels, decoder_channels):
        super().__init__()

        encoder_channels = encoder_channels[1:]  # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[::-1]  # reverse channels to start from head of encoder

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        seg_blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.seg_fusion_blocks = nn.ModuleList(seg_blocks[:-1])
        self.seg_final_block = seg_blocks[-1]

        dis_blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.dis_fusion_blocks = nn.ModuleList(dis_blocks[:-1])
        self.dis_final_block = dis_blocks[-1]

    def forward(self, *features):
        features = features[1:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        x_seg = features[0]
        x_dis = features[0]
        skips = features[1:]

        for i, decoder_block in enumerate(zip(self.seg_fusion_blocks, self.dis_fusion_blocks)):
            seg_decoder_block, dis_decoder_block = decoder_block
            skip = skips[i] if i < len(skips) else None
            x_seg_temp = seg_decoder_block(x_seg, skip)
            x_dis_temp = dis_decoder_block(x_dis, skip)

            x_seg = x_seg_temp + x_dis_temp
            x_dis = x_seg_temp + x_dis_temp

        x_seg = self.seg_final_block(x_seg, None)
        x_dis = self.dis_final_block(x_dis, None)

        return x_seg, x_dis


class Mynet(nn.Module):

    def __init__(self, num_classes=1, **kwargs):
        super().__init__()
        self.encoder = smp.Unet("resnet50", **kwargs).encoder
        self.decoder = UnetDecoder(self.encoder.out_channels, (256, 128, 64, 32, 16))

        self.seg_head = SegmentationHead(in_channels=16, out_channels=num_classes)
        self.dis_head = SegmentationHead(in_channels=16, out_channels=num_classes)

    def initialize(self):
        init.initialize_decoder(self.encoder)
        init.initialize_head(self.seg_head)
        init.initialize_head(self.dis_head)

    def forward(self, x):
        features = self.encoder(x)
        x_seg, x_dis = self.decoder(*features)

        x_seg = self.seg_head(x_seg)
        x_dis = self.dis_head(x_dis)

        return [x_seg, x_dis]
