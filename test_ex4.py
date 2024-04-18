import torch
from torchvision import transforms
from PIL import Image
from models.ops import ModelContainer
from utils.etc import AttnLabelConverter
from utils.dataset import AlignCollate
from cfg import cfg_cli
import string
import time

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

# Set the model to evaluation mode
model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define the path to the input image
image_path = opt.image_path  # Replace with the path to your input image

# Load the input image using PIL
image = Image.open(image_path)

image = image.convert('L')

# Define the image transformation using AlignCollate
align_collate = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)

# Transform the input image
transformed_image, _ = align_collate([(image, "")])  # The second argument is an empty string since you don't have labels


transformed_image = transformed_image.to(device)

# Perform inference (forward pass) on the input image
with torch.no_grad():  # No gradients needed for inference
    output, num_steps = model(transformed_image, is_train=False)
    infer_time = 0
    length_of_data = 0

    batch_size = transformed_image.size(0)
    length_of_data = length_of_data + batch_size
    image = transformed_image.to(device)
    # For max length prediction
    length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
    text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

    start_time = time.time()

    preds = model(image, text_for_pred, is_train=False)
    preds_tensor = preds[0]

    forward_time = time.time() - start_time

    preds_tensor = preds_tensor[:, :opt.batch_max_length, :]
    # select max probabilty (greedy decoding) then decode index to character
    _, preds_index = preds_tensor.max(2)

    preds_str = converter.decode(preds_index, length_for_pred)










# Print the recognized text
print('Recognized text:', preds_str[0])
