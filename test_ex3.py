import torch
from torchvision import transforms
from PIL import Image
from models.ops import ModelContainer
from utils.etc import AttnLabelConverter

from cfg import cfg_cli
import string
import math
from torch.utils.data import DataLoader

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
        batch = list(filter(lambda x: x is not None, batch))

        if self.keep_ratio_with_pad:  # same concept with 'Rosetta' paper
            resized_max_w = self.imgW
            input_channel = 3 if images[0].mode == 'RGB' else 1
            transform = NormalizePAD((input_channel, self.imgH, resized_max_w))

            resized_images = []

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
            image_tensors = transform(image)
            image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)

        return image_tensors



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the pre-trained model from file
opt = cfg_cli()
file_name = "./dict/ko.txt"
with open(file_name, encoding='utf-8') as fd:
    character = [char.strip('\n') for char in fd.readlines()]
    opt.character = ''.join(sorted(character)) + string.printable[:-6]  # same with ASTER setting (use)

model_path = opt.saved_model # Replace with the path to your trained model
# Ensure that input_size and num_embeddings are integers


converter = AttnLabelConverter(opt.character)  # Provide the characters set used for training
opt.num_class = len(converter.character)

model = ModelContainer(opt)  # Make sure you have the appropriate options (opt)
model = torch.nn.DataParallel(model).to(device)


loaded_model = torch.load(model_path)

if isinstance(loaded_model, torch.nn.DataParallel):
    # If the model is wrapped in DataParallel, unwrap it
    model = loaded_model.module
else:
    # If the model is not wrapped in DataParallel, assign it directly
    model = loaded_model


align_collate = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)



# Define the path to the input image
image_path = opt.image_path  # Replace with the path to your input image

# Load the input image using PIL
image = Image.open(image_path)

# Transform the input image
transformed_image = align_collate([(image, "")])  # The second argument is an empty string since you don't have labels


single_image_dataset = torch.utils.data.TensorDataset(transformed_image)

demo_loader = torch.utils.data.DataLoader(
    single_image_dataset, batch_size=opt.batch_size,  # Only one image in the batch
    shuffle=False,
    num_workers=int(opt.workers),
    collate_fn=align_collate,  # Use AlignCollate for proper alignment and transformation
    pin_memory=True
)

for image in demo_loader:
    print(image.shape)

# Set the model to evaluation mode
model.eval()
# Perform inference (forward pass) on the input image
with torch.no_grad():
    for image_tensors in demo_loader:
        batch_size = image_tensors.size(0)
        image = image_tensors.to(device)
        # For max length prediction
        length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
        text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

        preds = model(image, text_for_pred, is_train=False)

        # select max probabilty (greedy decoding) then decode index to character
        _, preds_index = preds.max(2)
        preds_str = converter.decode(preds_index, length_for_pred)





# Print the recognized text
print('Recognized text:', preds_str[0])
