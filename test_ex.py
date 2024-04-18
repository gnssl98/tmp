import math
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from cfg import cfg_cli
import torch.nn as nn
from models.backbone import VGG_FeatureExtractor
from models.sequence import BidirectionalLSTM
from models.prediction import Attention

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ModelContainer(nn.Module):
    def __init__(self, opt):
        super(ModelContainer, self).__init__()
        self.opt = opt
        # feature extraction
        self.FeatureExtraction = VGG_FeatureExtractor(opt.input_channel, opt.output_channel)
        self.FeatureExtraction_output = opt.output_channel
        # average pooling
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))
        # sequence modeling
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
            )
        )
        self.SequenceModeling_output = opt.hidden_size
        # prediction stage
        self.Prediction = Attention(
            self.SequenceModeling_output,
            opt.hidden_size, opt.num_class
        )

    def forward(self, input_x):
        # Perform feature extraction
        visual_feature_extraction = self.FeatureExtraction(input_x)
        # Reshape the feature extraction output
        visual_feature_extraction = self.AdaptiveAvgPool(
            visual_feature_extraction.permute(0, 3, 1, 2)
        )
        visual_feature_extraction = visual_feature_extraction.squeeze(3)

        # Perform sequence modeling
        context_feature = self.SequenceModeling(visual_feature_extraction)

        # Perform prediction
        decode = self.Prediction(context_feature.contiguous())

        return decode

class ResizeNormalize(object):

    def __init__(self, size, interpolation=Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


class NormalizePAD(object):

    def __init__(self, max_size, PAD_type='right'):
        self.toTensor = transforms.ToTensor()
        self.max_size = max_size
        self.max_width_half = math.floor(max_size[2] / 2)
        self.PAD_type = PAD_type

    def __call__(self, img):
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        c, h, w = img.size()
        Pad_img = torch.FloatTensor(*self.max_size).fill_(0)
        Pad_img[:, :, :w] = img  # right pad
        if self.max_size[2] != w:  # add border Pad
            Pad_img[:, :, w:] = img[:, :, w - 1].unsqueeze(2).expand(c, h, self.max_size[2] - w)

        return Pad_img



class AlignCollate(object):

    def __init__(self, imgH=32, imgW=100, keep_ratio_with_pad=False):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio_with_pad = keep_ratio_with_pad

    def __call__(self, batch):
        batch = filter(lambda x: x is not None, batch)
        images, labels = zip(*batch)

        if self.keep_ratio_with_pad:  # same concept with 'Rosetta' paper
            resized_max_w = self.imgW
            input_channel = 3 if images[0].mode == 'RGB' else 1
            transform = NormalizePAD((input_channel, self.imgH, resized_max_w))

            resized_images = []
            for image in images:
                w, h = image.size
                ratio = w / float(h)
                if math.ceil(self.imgH * ratio) > self.imgW:
                    resized_w = self.imgW
                else:
                    resized_w = math.ceil(self.imgH * ratio)

                resized_image = image.resize((resized_w, self.imgH), Image.BICUBIC)
                resized_images.append(transform(resized_image))
                # resized_image.save('./image_test/%d_test.jpg' % w)

            image_tensors = torch.cat([t.unsqueeze(0) for t in resized_images], 0)

        else:
            transform = ResizeNormalize((self.imgW, self.imgH))
            image_tensors = [transform(image) for image in images]
            image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)

        return image_tensors, labels


def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor.cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


# Converter
class AttnLabelConverter:
    """ Convert between text-label and text-index for inference. """

    def __init__(self, character):
        # Character (str): set of the possible characters.
        # Initialize with start and end tokens
        list_token = ['[GO]', '[s]']  # Start and end tokens
        list_character = list(character)
        self.character = list_token + list_character

        # Create a dictionary mapping characters to indices
        self.dict = {char: i for i, char in enumerate(self.character)}

    def decode(self, text_index, length):
        """ Convert text-index into text-label.

        Args:
            text_index: Tensor containing the text indices (output from the model).
            length: List containing the length of each text label.

        Returns:
            List of decoded text labels.
        """
        texts = []
        for index, l in enumerate(length):
            # Decode text indices to characters and join them to form a string
            text = ''.join([self.character[i] for i in text_index[index, :]])
            # Remove padding based on the length provided
            texts.append(text[:l])
        return texts


if __name__ == '__main__':
    # Model initialization
    opt = cfg_cli()
    model = ModelContainer(opt)

    # Load the trained model from a file
    model_path = opt.saved_model
    if model_path == '':
        raise ValueError("Model path not specified.")

    # Load the model state
    model.load_state_dict(torch.load(model_path))

    # Set the model to evaluation mode
    model.eval()

    # Load the image
    image = Image.open(opt.image_path)

    # Prepare image transformations
    align_collate = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)

    # Transform the image and convert it to a tensor
    transformed_image, _ = align_collate([(image, '')])

    # Move the image tensor to the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transformed_image = transformed_image.to(device)

    # Move the model to the device
    model.to(device)

    # Perform inference: forward pass the input image through the model
    with torch.no_grad():  # No gradients needed for inference
        # Add batch dimension to the input tensor and pass it to the model
        output = model(transformed_image)

    # Assuming `output` is the text indices tensor and `lengths` is a list of lengths
    # You need to extract these from your model output accordingly
    # If your model returns predictions differently, adjust this accordingly
    preds = output
    lengths = [len(preds)]  # Adjust to obtain the correct lengths

    # Convert predictions to text labels using the converter
    converter = AttnLabelConverter(opt.character)
    decoded_texts = converter.decode(preds, lengths)

    # Print the decoded text labels
    for text in decoded_texts:
        print('Predicted text:', text)

'''
if __name__ == '__main__':

# modelcontainer
    opt = cfg_cli()
    model = ModelContainer(opt)

# Load the trained model from a file
    model_path = opt.saved_model
    if model_path == '':
        raise ValueError

    model.load(torch.load(model_path))

# Set the model to evaluation mode
    model.eval()
    # Define the path to the input image
    image = Image.open(opt.image_path)
    align_collate = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    transformed_image = align_collate(image)

    converter = AttnLabelConverter(opt.character)
    text, length = converter.decode(output, lengths)
    preds = model(image, text[:, :-1])  # align with Attention.forward
    target = text[:, 1:]  # without [GO] Symbol

    # Move the image tensor to the same device as the model (e.g., CUDA)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_tensor = transformed_image.to(device)

    # Move the model to the device
    model.to(device)

    # Perform inference: forward pass the input image through the model
    with torch.no_grad():  # No gradients needed for inference
        output = model(image_tensor.unsqueeze(0))  # Add batch dimension to the input tensor

    # Interpret the output as needed (depends on your model's task and output format)
    # For example, if the model is for classification, you can interpret the output as class predictions:
    _, predicted_class = torch.max(output, 1)

    # Print the model's predicted class or output
    print('Predicted class:', predicted_class.item())

'''