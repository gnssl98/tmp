import torch.nn as nn
from .backbone import VGG_FeatureExtractor
from .sequence import BidirectionalLSTM
from .prediction import Attention


class ModelContainer(nn.Module):

    def __init__(self, opt):
        super(ModelContainer, self).__init__()
        self.opt = opt
        # feature extraction
        self.FeatureExtraction = VGG_FeatureExtractor(opt.input_channel, opt.output_channel)
        # ing (image height / 16 - 1) * 15
        self.FeatureExtraction_output = opt.output_channel
        # fully connected layer (x)
        # average pooling (image height / 16 - 1) -> 1
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))
        self.SequenceModeling = nn.Sequential(
            BidirectionalLSTM(
                self.FeatureExtraction_output,
                opt.hidden_size,
                opt.hidden_size
            ),
            BidirectionalLSTM(
                opt.hidden_size,
                opt.hidden_size,
                opt.hidden_size
            ))
        self.SequenceModeling_output = opt.hidden_size
        # decode stage
        self.Prediction = Attention(
            self.SequenceModeling_output,
            opt.hidden_size, opt.num_class
        )

    def forward(self, input_x, label=None, is_train=True):
        # Extract visual features
        visual_feature_extraction = self.FeatureExtraction(input_x)

        # Reshape the output of feature extraction
        visual_feature_extraction = self.AdaptiveAvgPool(
            visual_feature_extraction.permute(0, 3, 1, 2)
        )
        visual_feature_extraction = visual_feature_extraction.squeeze(3)

        # Apply sequence modeling
        context_feature = self.SequenceModeling(visual_feature_extraction)

        # Perform prediction based on context features
        decode = self.Prediction(
            context_feature.contiguous(),
            label,
            is_train,
            batch_max_length=self.opt.batch_max_length
        )

        return decode



'''
#원본코드
    def forward(self, input_x, label, is_train=True):
        visual_feature_extraction = self.FeatureExtraction(input_x)
        # reshape [batch, channel, height, width] -> [batch, width, channel, height]
        visual_feature_extraction = self.AdaptiveAvgPool(visual_feature_extraction.permute(0, 3, 1, 2))
        visual_feature_extraction = visual_feature_extraction.squeeze(3)

        context_feature = self.SequenceModeling(visual_feature_extraction)
        decode = self.Prediction(context_feature.contiguous(), label, is_train,
                                 batch_max_length=self.opt.batch_max_length)

        return decode
'''
