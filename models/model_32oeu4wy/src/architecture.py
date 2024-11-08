import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

########################################################################################
# RNN Model by Parsanna Koirala :)

# ======================================================================================== #

# a custom convolutional LSTM that maintains spatial informtion while processing temporal data
class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell, self).__init__()\
          
        # initializing the convLSTM cells with input channels, hidden channels, 
        # and convolutional kernel size (kernel)
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        # single convolution operation that computes all gates at once
        # output channels = 4 * hidden channels for input, forgot, output, and cell
        self.conv = nn.Conv2d (
            in_channels = input_channels + hidden_channels,
            out_channels = 4 * hidden_channels,
            kernel_size = kernel_size,
            padding = self.padding,
            bias = True
        )
        
    # forward pass of the convLSTM cell
    # returns a tuple (new hidden state, new cell state) for the next timestep
    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim = 1)

        conv_output = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(conv_output, self.hidden_channels, dim = 1)

        # applying activation functions to compute gate values
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        
        # updating cell state using forget and input gates
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next
    
# ======================================================================================== #

# implementing a downsampling path with skip connections using double convolution with batch normalization,
# relu activation, and max pooling
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        # first convolution layer with batch normalization
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1)
        self.bn1 = nn.BatchNorm2d(out_channels) # BatchNorm2d normalizes output to prevent internal covariate shifts
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # relu activation function
        self.relu = nn.ReLU(inplace = True) # inplace saves memory
        
        # max pooling layer for downsampling; the MaxPool2d function reduces spatial dimensions by half
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)

    def forward(self, x):
      x = self.relu(self.bn1(self.conv1(x)))
      x = self.relu(self.bn2(self.conv2(x)))

      # storing core features before pooling for skip connection
      skip = x
      
      # downsampling using max pooling
      x = self.pool(x)
      return x, skip

# ======================================================================================== #

# implementing a upsampling path using transposed convolution for upsampling, skip connection concatenation,
# and double convollution with batch normalization
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()

        # transposed convolution for upsampling
        # ConvTranspose2d doubles the spatial dimensions and halves the number of channels
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size = 2, stride = 2)
        
        # first convolution after 
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # relu activation function
        self.relu = nn.ReLU(inplace = True) 
        
    def forward(self, x, skip):
      
      # upsampling the input and then concatenating with skip connection along the channel dimension
      x = self.upsample(x)
      x = torch.cat([x, skip], dim = 1)
      
      x = self.relu(self.bn1(self.conv1(x)))
      x = self.relu(self.bn2(self.conv2(x)))
      
      return x

# ======================================================================================== #

# the scene perception architecture takes in multiple frames as input based on the lookback configs,
# uses an encoder-decoder architecture with skip connections, uses ConvLSTM for temporal processing,
# and outputs segmentation maps with the background, cones, driveable area, and lane lines
class scene_perception_model(nn.Module):
  def __init__(self, lookback, input_channels = 3, num_classes = 4):
    super(scene_perception_model, self).__init__()
    
    # calculating the total input channels needed based on lookback
    self.total_input_channels = input_channels * (lookback['count'] + 1)
    
    # encoder
    self.enc1 = EncoderBlock(self.total_input_channels, 64)
    self.enc2 = EncoderBlock(64, 128)
    self.enc3 = EncoderBlock(128, 256)
    self.enc4 = EncoderBlock(256, 512)
    
    # bridge
    self.bridge_conv = nn.Conv2d(512, 1024, kernel_size = 3, padding = 1)
    self.bridge_bn = nn.BatchNorm2d(1024)
    
    # calling convLSTM for temoporal processing
    self.convlstm = ConvLSTMCell(1024, 1024, kernel_size = 3)
    
    # decoder
    self.dec4 = DecoderBlock(1024, 512)
    self.dec3 = DecoderBlock(512, 256)
    self.dec2 = DecoderBlock(256, 128)
    self.dec1 = DecoderBlock(128, 64)
    
    # final classification
    self.final_conv = nn.Conv2d(64, num_classes, kernel_size = 1)
    self.relu = nn.ReLU(inplace = True)
    
  def forward(self, x):
    batch_size = x.size(0)
    
    # encoder path with skip connections
    x1, skip1 = self.enc1(x)
    x2, skip2 = self.enc2(x1)
    x3, skip3 = self.enc3(x2)
    x4, skip4 = self.enc4(x3)
    
    # bridge
    x = self.relu(self.bridge_bn(self.bridge_conv(x4)))
    
    # initializing the ConvLSTM state
    h = torch.zeros(batch_size, 1024, x.size(2), x.size(3)).to(x.device)
    c = torch.zeros(batch_size, 1024, x.size(2), x.size(3)).to(x.device)
    
    # processing ConvLSTM
    h, c = self.convlstm(x, (h, c))
    
    # decoder path
    x = self.dec4(h, skip4)
    x = self.dec3(x, skip3)
    x = self.dec2(x, skip2)
    x = self.dec1(x, skip1)
    
    # final classification
    x = self.final_conv(x)
    
    return x
  
  def init_weights(self):
    for m in self.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

# ======================================================================================== #
