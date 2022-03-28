import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.unet.decoder import DecoderBlock
from segmentation_models_pytorch.base import SegmentationHead
from segmentation_models_pytorch.base import initialization as init


def add_conv(in_ch, out_ch, ksize, stride):
    """
    Add a conv2d / batchnorm / leaky ReLU block.
    Args:
        in_ch (int): number of input channels of the convolution layer.
        out_ch (int): number of output channels of the convolution layer.
        ksize (int): kernel size of the convolution layer.
        stride (int): stride of the convolution layer.
    Returns:
        stage (Sequential) : Sequential layers composing a convolution block.
    """
    stage = nn.Sequential()
    pad = (ksize - 1) // 2
    stage.add_module('conv', nn.Conv2d(in_channels=in_ch,
                                       out_channels=out_ch, kernel_size=ksize, stride=stride,
                                       padding=pad, bias=False))
    stage.add_module('batch_norm', nn.BatchNorm2d(out_ch))
    stage.add_module('relu', nn.ReLU(inplace=True))
    return stage


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

        self.initialize()

    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.seg_head)
        init.initialize_head(self.dis_head)

    def forward(self, x):
        features = self.encoder(x)
        x_seg, x_dis = self.decoder(*features)

        x_seg = self.seg_head(x_seg)
        x_dis = self.dis_head(x_dis)

        return [x_seg, x_dis]


class UnetDecoderWithoutFusion(nn.Module):

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
        self.seg_blocks = nn.ModuleList(seg_blocks)

        dis_blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.dis_blocks = nn.ModuleList(dis_blocks)

    def forward(self, *features):
        features = features[1:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        x_seg = features[0]
        x_dis = features[0]
        skips = features[1:]

        for i, decoder_block in enumerate(zip(self.seg_blocks, self.dis_blocks)):
            seg_decoder_block, dis_decoder_block = decoder_block
            skip = skips[i] if i < len(skips) else None
            x_seg = seg_decoder_block(x_seg, skip)
            x_dis = dis_decoder_block(x_dis, skip)

        return x_seg, x_dis


class MynetWithoutFusion(nn.Module):

    def __init__(self, num_classes=1, **kwargs):
        super().__init__()
        self.encoder = smp.Unet("resnet50", **kwargs).encoder
        self.decoder = UnetDecoderWithoutFusion(self.encoder.out_channels, (256, 128, 64, 32, 16))

        self.seg_head = SegmentationHead(in_channels=16, out_channels=num_classes)
        self.dis_head = SegmentationHead(in_channels=16, out_channels=num_classes)

        self.initialize()

    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.seg_head)
        init.initialize_head(self.dis_head)

    def forward(self, x):
        features = self.encoder(x)
        x_seg, x_dis = self.decoder(*features)

        x_seg = self.seg_head(x_seg)
        x_dis = self.dis_head(x_dis)

        return [x_seg, x_dis]


class WeightedFusionModule(nn.Module):

    def __init__(self, in_channel):
        super().__init__()
        self.compress_seg = add_conv(in_channel, 16, 1, 1)
        self.compress_dis = add_conv(in_channel, 16, 1, 1)

        self.weight_seg = nn.Conv2d(32, 2, 1, 1, 0)
        self.weight_dis = nn.Conv2d(32, 2, 1, 1, 0)
        
    def forward(self, x_seg, x_dis):
        x_seg_compressed = self.compress_seg(x_seg)
        x_dis_compressed = self.compress_dis(x_dis)

        x_compressed = torch.cat((x_seg_compressed, x_dis_compressed), 1)

        seg_weight = self.weight_seg(x_compressed)
        seg_weight = F.softmax(seg_weight, dim=1)

        dis_weight = self.weight_dis(x_compressed)
        dis_weight = F.softmax(dis_weight, dim=1)

        x_seg_fusion = x_seg * seg_weight[:, 0:1, :, :] + x_dis * seg_weight[:, 1:, :, :]
        x_dis_fusion = x_seg * dis_weight[:, 0:1, :, :] + x_dis * dis_weight[:, 1:, :, :]

        return x_seg_fusion, x_dis_fusion


class OnlyOneBranchWeightedFusionModule(nn.Module):
    def __init__(self, in_channel, fusion_branch='seg'):
        super().__init__()
        self.fusion_branch = fusion_branch
        self.compress_seg = add_conv(in_channel, 16, 1, 1)
        self.compress_dis = add_conv(in_channel, 16, 1, 1)

        self.weight_feature = nn.Conv2d(32, 2, 1, 1, 0)

    def forward(self, x_seg, x_dis):
        x_seg_compressed = self.compress_seg(x_seg)
        x_dis_compressed = self.compress_dis(x_dis)

        x_compressed = torch.cat((x_seg_compressed, x_dis_compressed), 1)

        weight = self.weight_feature(x_compressed)

        if self.fusion_branch == 'seg':
            x_seg_fusion = x_seg * weight[:, 0:1, :, :] + x_dis * weight[:, 1:, :, :]
            x_dis_fusion = x_dis
        elif self.fusion_branch == 'dis':
            x_seg_fusion = x_seg
            x_dis_fusion = x_seg * weight[:, 0:1, :, :] + x_dis * weight[:, 1:, :, :]

        return x_seg_fusion, x_dis_fusion

class UnetDecoderWeightedFusion(nn.Module):

    def __init__(self, encoder_channels, decoder_channels):
        super().__init__()

        encoder_channels = encoder_channels[1:]  # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[::-1]  # reverse channels to start from head of encoder

        # computing blocks input and output channels
        head_channels = encoder_channels[0]  # 2048
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

        weight_in_channels = decoder_channels[:-1]
        weight_blocks = [
            WeightedFusionModule(in_ch) for in_ch in weight_in_channels
        ]
        self.weight_blocks = nn.ModuleList(weight_blocks)


    def forward(self, *features):
        features = features[1:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        x_seg = features[0]
        x_dis = features[0]
        skips = features[1:]

        for i, decoder_block in enumerate(zip(self.seg_fusion_blocks, self.dis_fusion_blocks, self.weight_blocks)):
            seg_decoder_block, dis_decoder_block, weight_block = decoder_block
            skip = skips[i] if i < len(skips) else None
            x_seg = seg_decoder_block(x_seg, skip)
            x_dis = dis_decoder_block(x_dis, skip)

            x_seg, x_dis = weight_block(x_seg, x_dis)

        x_seg = self.seg_final_block(x_seg, None)
        x_dis = self.dis_final_block(x_dis, None)

        return x_seg, x_dis


class MynetWeightedFusion(nn.Module):

    def __init__(self, num_classes=1, **kwargs):
        super().__init__()
        self.encoder = smp.Unet("resnet50", **kwargs).encoder
        self.decoder = UnetDecoderWeightedFusion(self.encoder.out_channels, (256, 128, 64, 32, 16))

        self.seg_head = SegmentationHead(in_channels=16, out_channels=num_classes)
        self.dis_head = SegmentationHead(in_channels=16, out_channels=num_classes)

        self.initialize()

    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.seg_head)
        init.initialize_head(self.dis_head)

    def forward(self, x):
        features = self.encoder(x)
        x_seg, x_dis = self.decoder(*features)

        x_seg = self.seg_head(x_seg)
        x_dis = self.dis_head(x_dis)

        return [x_seg, x_dis]


class UnetDecoderOnlySegFusion(UnetDecoderWeightedFusion):

    def __init__(self, encoder_channels, decoder_channels):
        super().__init__(encoder_channels, decoder_channels)

        weight_in_channels = decoder_channels[:-1]
        weight_blocks = [
            OnlyOneBranchWeightedFusionModule(in_ch) for in_ch in weight_in_channels
        ]
        self.weight_blocks = nn.ModuleList(weight_blocks)


class MynetOnlySegFusion(Mynet):

    def __init__(self, num_classes=1, **kwargs):
        super().__init__()
        self.encoder = smp.Unet("resnet50", **kwargs).encoder
        self.decoder = UnetDecoderOnlySegFusion(self.encoder.out_channels, (256, 128, 64, 32, 16))

        self.seg_head = SegmentationHead(in_channels=16, out_channels=num_classes)
        self.dis_head = SegmentationHead(in_channels=16, out_channels=num_classes)

        self.initialize()
