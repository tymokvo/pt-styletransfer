"""
perform neural style transfer and display the result.

To Do:
    Add live visualization
"""
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

import time
from time import sleep
from time import time as now
#import PIL
from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms

from sys import exit
import os

from vgg import vgg19

from collections import OrderedDict


def get_date_time():
    """return a tuple in the format (d-m-y, h-m-s)"""
    dmy = time.strftime('%d-%m-%y')
    hms = time.strftime('%H:%M:%S')
    return dmy, hms


imsize = 400  # desired size of the output image, 1120 working on 8gb gtx 1080 (uses ~6.9gb of vram)

loader = transforms.Compose([transforms.Scale(imsize), transforms.ToTensor()])  # scale imported image and transform it into a torch tensor

pictures_dir = '/home/tyler/Pictures/'  #put a directory full of images here

#convenience for selecting images
micromegas = 'micromegas_square.jpg'
acropolis = 'acropolis_square_large.jpg'

date1, _ = get_date_time()

save_dir = pictures_dir + 'st_output_{}/'.format(date1)

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

def image_loader(image_name):
    """load image"""
    image = Image.open(image_name)
    image = Variable(loader(image))
    image = image.unsqueeze(0)  # fake batch dimension required to fit network's input dimensions
    return image

style_image = pictures_dir + micromegas
content_image = pictures_dir + acropolis

style = image_loader(style_image).cuda()
content = image_loader(content_image).cuda()

style_name = style_image.split('/')[-1]
content_name = content_image.split('/')[-1]

assert style.data.size() == content.data.size(), "we need to import style and content images of the same size"


########## display

unloader = transforms.ToPILImage()  # reconvert into PIL image


def imshow(tensor):
    """show an image"""
    image = tensor.clone().cpu()  # we clone the tensor in order to not do changes on it
    image.resize_(3, imsize, imsize)  # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)


def tensor_to_image(tensor):
    """show an image"""
    image = tensor.clone().cpu()  # we clone the tensor in order to not do changes on it
    image.resize_(3, imsize, imsize)  # remove the fake batch dimension
    image = unloader(image)
    return image


def display_correction(input):
    """correct for display"""
    result = input.data.cpu().clone()
    result = result.numpy()
    result[result < 0] = 0
    result[result > 1] = 1
    result = torch.from_numpy(result)
    return result


def tensors_to_imgs(activations, save_dir, colormap='magma'):
    """save a list of activation tensors to disk as images"""
    date, time = get_date_time()
    for tensor in activations:
        layer_tag = 0
        for i in range(tensor.size()[1] - 1):
            filename = '{}_{}_conv_{}_{}'.format(date, time, layer_tag, i)
            tensor_np = convert_display_Ttensor(tensor[i, :, :])
            if i % 20 == 0:
                sleep(.1)
            plt.imsave(save_dir + filename, tensor_np, cmap=colormap)

        layer_tag += 1


#------------
"""define classes"""
#------------


class ContentLoss(nn.Module):
    """Calculate the loss of the content layers"""

    def __init__(self, target, weight):
        """initialize"""
        super(ContentLoss, self).__init__()
        self.target = target.detach() * weight
        """
        we 'detach' the target content from the tree used to dynamically compute the gradient:
        this is a stated value, not a variable.
        Otherwise the forward method of the criterion will throw an error.
        """
        self.weight = weight
        self.criterion = nn.MSELoss()

    def forward(self, input):
        """compute forward pass"""
        self.loss = self.criterion.forward(input*self.weight, self.target)
        self.output = input
        return self.output

    def backward(self, retain_variables=True):
        """compute backward pass"""
        self.loss.backward(retain_variables=retain_variables)
        return self.loss.data[0]


class GramMatrix(nn.Module):
    """Calculate a gram matrix"""

    def __init__(self):
        super(GramMatrix, self).__init__()
        self.record = None

    def forward(self, input):
        """compute forward pass"""
        a, b, c, d = input.data.size()  # a=batch size(=1) || b=number of feature maps || (c,d)=dimensions of a f. map (N=c*d)

        input.data.resize_(a*b, c*d)  # resise F_XL into \hat F_XL

        G = torch.mm(input, input.t())  # compute the gram product
        G.div_(a*b*c*d)  # we 'normalize' the values of the gram matrix by dividing by the number of element in each feature maps.

        self.record = G.data.clone().cpu()

        return G


class StyleLoss(nn.Module):
    """calculate a style loss"""

    def __init__(self, target, strength):
        """initialize"""
        super(StyleLoss, self).__init__()
        self.target = target.detach()*strength
        self.strength = strength
        self.gram = GramMatrix()
        self.criterion = nn.MSELoss()
        self.gram_record = None  # this is will become a global list out from iterations of GramMatrix, will be overwritten at every forward

    def forward(self, input):
        """compute forward pass"""
        self.output = input.clone()
        self.G = self.gram.forward(input)
        self.G.mul_(self.strength)
        self.loss = self.criterion.forward(self.G, self.target)
        self.gram_record = self.gram.record
        return self.output

    def backward(self, retain_variables=True):
        """compute backward pass"""
        self.loss.backward(retain_variables=retain_variables)
        return self.loss.data[0]

####### load model

#cnn = models.alexnet(pretrained=True).features.cuda()  # Alexnet has 5 Conv2d layers
vgg_19 = vgg19(pretrained=True).features.cuda()  # VGG19 has 16 Conv2d layers
cnn = vgg_19

# desired depth layers to compute style/content losses :
content_layers = ['conv_10', 'conv_13', 'conv_15']
style_layers = ['conv_1', 'conv_3', 'conv_5', 'conv_7']


#------------
"""make StyleTransferNet class"""
#------------

4
class StyleTransferNet(nn.Module):
    """a network for neural style transfer"""

    def __init__(self, submodule, content_layers, style_layers):
        """submodule works with the features submodule of VGG
        content_layers and style_layers should be list of strings like:
        'conv_1'

        """
        super(StyleTransferNet, self).__init__()
        self.ArtNet = nn.Sequential().cuda()
        self.submodule = submodule  # an existing network like VGG
        self.gram = GramMatrix()
        self._content_layers = content_layers
        self._style_layers = style_layers
        self._unbuilt = True  # a better name might be 'under construction'
        #these will be publicly available
        self.content_losses = []
        self.style_losses = []
        self.style_targets = []
        self.content_targets = []
        self.grams = None  # this is pulling a list out of StyleLoss

    def build_network(self, content_img, style_img, content_weight, style_weight, save_trigger=False):  # TODO: change this to be stateless
        """construct network, add gram matricies and content/style losses
        save_trigger allows extracting intermediate representations
        """
        n_conv = 0
        n_relu = 0
        n_pool = 0
        for key, layer in self.submodule._modules.items():
            #iterate through dicitonary of pretrained modules1
            conv_name = 'conv_' + str(n_conv)
            relu_name = 'relu_' + str(n_relu)
            pool_name = 'pool_' + str(n_pool)
            if isinstance(layer, torch.nn.modules.conv.Conv2d):
                self.ArtNet.add_module(conv_name, layer)
                if conv_name in self._content_layers:
                    target = self.ArtNet.forward(content_img.cuda()).clone()
                    if save_trigger:
                        self.content_targets += [target.data.cpu()]
                    content_loss = ContentLoss(target, content_weight).cuda()
                    self.ArtNet.add_module('content_loss_' + str(n_conv), content_loss)
                    if self._unbuilt:
                        self.content_losses.append(content_loss)

                if conv_name in self._style_layers:
                    target_feature = self.ArtNet.forward(style_img.cuda()).clone()
                    if save_trigger:
                        self.style_targets += [target_feature.data.cpu()]
                    target_feature_gram = self.gram.forward(target_feature)
                    style_loss = StyleLoss(target_feature_gram, style_weight).cuda()
                    self.ArtNet.add_module('style_loss_' + str(n_conv), style_loss)
                    if self._unbuilt:
                        self.style_losses.append(style_loss)
                n_conv += 1

            if isinstance(layer, torch.nn.modules.activation.ReLU):
                self.ArtNet.add_module(relu_name, layer)
                if relu_name in self._content_layers:
                    target = self.ArtNet.forward(content_img.cuda()).clone()
                    if save_trigger:
                        self.content_targets += [target.data.cpu()]
                    content_loss = ContentLoss(target, content_weight).cuda()
                    self.ArtNet.add_module("content_loss_" + str(n_relu), content_loss)
                    if self._unbuilt:
                        self.content_losses.append(content_loss)

                if relu_name in self._style_layers:
                    target_feature = self.ArtNet.forward(style_img.cuda()).clone()
                    if save_trigger:
                        self.style_targets += [target_feature.data.cpu()]
                    target_feature_gram = self.gram.forward(target_feature)
                    style_loss = StyleLoss(target_feature_gram, style_weight).cuda()
                    self.ArtNet.add_module("style_loss_" + str(n_relu), style_loss)
                    if self._unbuilt:
                        self.style_losses.append(style_loss)
                n_relu += 1

            if isinstance(layer, torch.nn.modules.pooling.MaxPool2d):
                self.ArtNet.add_module(pool_name, layer)
                n_pool += 1
        self._unbuilt = False
        return

    def clear_network(self):
        """reset ArtNet to empty nn.Sequential"""
        self.ArtNet = nn.Sequential.cuda()

content_weight = 1
style_weight = 500

StyleNet = StyleTransferNet(cnn, content_layers, style_layers)
StyleNet.build_network(content, style, content_weight, style_weight, save_trigger=False)


###### input image

input = image_loader(content_image).cuda()
# if we want to fill it with a white noise:
#input.data = torch.randn(input.data.size()).cuda()

######## gradient descent

input = nn.Parameter(input.data)  # this line to show that input is a parameter that requires a gradient

learning_rate = 0.1

lr = np.linspace(0.01, 0.5, 20)

optimizer = optim.Adam([input], lr=learning_rate)
optimizer = optim.LBFGS([input], lr=learning_rate)

style_graph = []
content_graph = []

num_runs = 200

hyperparameters = {'learning_rate': learning_rate, 'iterations': num_runs, 'style_weight': style_weight, 'content_weight': content_weight, 'style_layers': style_layers, 'content_layers': content_layers}
hyperparameters = OrderedDict(sorted(hyperparameters.items()))  # make a sorted dictionary

grams = []
print('running...')
t1 = now()
for run in range(num_runs):

    # correct the values of updated input image
    updated = input.data.cpu().clone()
    updated = updated.numpy()
    updated[updated < 0] = 0
    updated[updated > 1] = 1
    input.data = torch.from_numpy(updated).cuda()

    optimizer.zero_grad()
    StyleNet.ArtNet.forward(input)
    style_score = 0
    content_score = 0

    """
    #this can be used to view gram matricies
    for key, module in StyleNet.ArtNet._modules.items():
        if isinstance(module, StyleLoss):
            grams += [module.gram_record]  #this is going to append quite a large list at every forward pass
    """

    for sl in StyleNet.style_losses:
        style_score += sl.backward()
    for cl in StyleNet.content_losses:
        content_score += cl.backward()

    optimizer.step()

    if run % 10 == 0:
        print("run: {} style: {} content: {}".format(run, style_score, content_score))

    style_graph.append(style_score)
    content_graph.append(content_score)
t2 = now()
print('{} runs in {} seconds at {}x{}px'.format(num_runs, t2 - t1, imsize, imsize))
print('grams: {}'.format(len(grams)))

#for i, gram in enumerate(grams):
    #grams[i] = grams[i].numpy()

#show(grams[0:9], grid_dims='auto', cmap='gnuplot2')
#exit()

style_graph = np.array(style_graph)
content_graph = np.array(content_graph)

# a last correction...
result = input.data.cpu().clone()
result = result.numpy()
result[result < 0] = 0
result[result > 1] = 1
result = torch.from_numpy(result)

# always save the output with the date and time appended
output_image = tensor_to_image(result)
date, time = get_date_time()
filename = save_dir + date + '_' + time
output = filename + '_0_output' + '.png'
output_image.save(output)

#save style and content images back to disk
style_out = tensor_to_image(style.data.cpu().clone())
content_out = tensor_to_image(content.data.cpu().clone())

style_out.save(filename + '_1_style' + '.png')
content_out.save(filename + '_2_content' + '.png')


def get_max_min(array):
    """get max and min for graph annotation. returns (x_max, y_max), (x_min, y_min)"""
    x_max = np.argmax(array)
    y_max = array[x_max]
    x_min = np.argmin(array)
    y_min = array[x_min]
    return (x_max, y_max), (x_min, y_min)

xy_shift = num_runs / 10

fig = plt.figure()
fig.set_size_inches(12, 5)
fig.suptitle('image: {}'.format(output.split('/')[-1]))
ax = fig.add_subplot(121)
fig.subplots_adjust(top=0.85)

#ax.set_title('learning rate: {}'.format(learning_rate))
ax.plot(style_graph, label='style loss')
ax.plot(content_graph, label='content loss')

style_max, style_min = get_max_min(style_graph)
content_graph[0] = content_graph[1]  # eliminate zero in first run for max/min annotations
content_max, content_min = get_max_min(content_graph)

#max/min plotting
ax.scatter(style_max[0], style_max[1], c='blue')
plt.text(style_max[0], style_max[1], str(style_max[1]))
ax.scatter(style_min[0], style_min[1], c='blue')
plt.text(style_min[0], style_min[1] + xy_shift, str(style_min[1]))
ax.scatter(content_max[0] + 1, content_max[1], c='orange')  # adjust x for 0 removal above
plt.text(content_max[0], content_max[1], str(content_max[1]))
ax.scatter(content_min[0], content_min[1], c='orange')
plt.text(content_min[0], content_min[1] - xy_shift, str(content_min[1]))


ax.set_xlabel('# iterations')
ax.set_ylabel('global mean loss')
ax.legend()

desc = fig.add_subplot(122)
desc.axis('off')
lim = len(hyperparameters)
desc.set_ylim(0, lim)
desc.set_xlim(0, lim)
i = 0
for k, v in hyperparameters.items():
    desc.text(2, i, k + ': ', ha='right')
    desc.text(2, i, v, ha='left', color='blue')
    i += 1

plt.savefig(save_dir + date + '_' + time + '_graph' + '.png', dpi=150)
