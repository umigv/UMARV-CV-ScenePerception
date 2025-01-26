import torch
import torch.nn as nn
import torch.nn.functional as F

########################################################################################
# RNN Model by Parsanna Koirala

# ======================================================================================== #

# a custom convolutional LSTM that maintains spatial informtion while processing temporal data
class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell, self).__init__()
          
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
# and double convolution with batch normalization
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
    def __init__(self, lookback, input_channels=3, num_classes=4):
        super(scene_perception_model, self).__init__()
        
        # First convolution takes 3 channels, not total_input_channels
        self.enc1 = EncoderBlock(input_channels, 64)
        self.enc2 = EncoderBlock(64, 128)
        self.enc3 = EncoderBlock(128, 256)
        self.enc4 = EncoderBlock(256, 512)
        
        # Bridge
        self.bridge_conv = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.bridge_bn = nn.BatchNorm2d(1024)
        
        # ConvLSTM for temporal processing
        self.convlstm = ConvLSTMCell(1024, 1024, kernel_size=3)
        
        # Decoder
        self.dec4 = DecoderBlock(1024, 512)
        self.dec3 = DecoderBlock(512, 256)
        self.dec2 = DecoderBlock(256, 128)
        self.dec1 = DecoderBlock(128, 64)
        
        # Final classification
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x shape: [batch_size, num_frames, channels, height, width]
        batch_size = x.size(0)
        num_frames = x.size(1)
        
        # Process each frame through encoder
        encoded_frames = []
        last_skip_connections = None
        
        for t in range(num_frames):
            curr_frame = x[:, t]  # [batch_size, channels, height, width]
            
            # Encoder path
            x1, skip1 = self.enc1(curr_frame)
            x2, skip2 = self.enc2(x1)
            x3, skip3 = self.enc3(x2)
            x4, skip4 = self.enc4(x3)
            
            # Bridge
            encoded = self.relu(self.bridge_bn(self.bridge_conv(x4)))
            encoded_frames.append(encoded)
            
            # Store skip connections from last frame
            last_skip_connections = (skip1, skip2, skip3, skip4)
        
        # Initialize ConvLSTM state
        h = torch.zeros(batch_size, 1024, encoded.size(2), encoded.size(3)).to(x.device)
        c = torch.zeros(batch_size, 1024, encoded.size(2), encoded.size(3)).to(x.device)
        
        # Process through ConvLSTM
        for encoded in encoded_frames:
            h, c = self.convlstm(encoded, (h, c))
        
        # Use last frame's skip connections for decoder
        skip1, skip2, skip3, skip4 = last_skip_connections
        
        # Decoder path
        x = self.dec4(h, skip4)
        x = self.dec3(x, skip3)
        x = self.dec2(x, skip2)
        x = self.dec1(x, skip1)
        
        # Final classification
        x = self.final_conv(x)
        
        return x

# ======================================================================================== #

"""
# Basic CNN model for scene perception
# class scene_perception_model(nn.Module):
#   def __init__(self, lookback):
#     super(scene_perception_model, self).__init__()
#     self.model = nn.Sequential(
#       nn.Conv2d( in_channels=3+3*lookback['count'], out_channels=20 , kernel_size=15 , padding=7 , stride=1 ),
#       nn.BatchNorm2d(20),
#       nn.LeakyReLU(),
#       nn.Conv2d( in_channels=20 , out_channels=20 , kernel_size=15 , padding=7 , stride=1 ),
#       nn.BatchNorm2d(20),
#       nn.LeakyReLU(),
#       nn.Conv2d( in_channels=20 , out_channels=20 , kernel_size=15 , padding=7 , stride=1 ),
#       nn.BatchNorm2d(20),
#       nn.LeakyReLU(),
#       nn.Conv2d( in_channels=20 , out_channels=10 , kernel_size=15 , padding=7 , stride=1 ),
#       nn.BatchNorm2d(10),
#       nn.LeakyReLU(),
#       nn.Conv2d( in_channels=10 , out_channels=4 , kernel_size=15 , padding=7 , stride=1 ),
#     )
#   def forward(self, input):
#     output = self.model(input)
#     return output

# Basic U-Net model for scene perception
class scene_perception_model(nn.Module):

  def __init__(self, lookback):
    super(scene_perception_model, self).__init__()

    in_channels = 3 + 3 * lookback['count']
    out_channels = 4

    # Downsampling
    self.enc1 = self.conv_block(in_channels, 64)
    self.enc2 = self.conv_block(64, 128)
    self.enc3 = self.conv_block(128, 256)
    self.enc4 = self.conv_block(256, 512)
    
    # Bottleneck
    self.bottleneck = self.conv_block(512, 1024)
    
    # Upsampling
    self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
    self.dec4 = self.conv_block(1024, 512)
    self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
    self.dec3 = self.conv_block(512, 256)
    self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
    self.dec2 = self.conv_block(256, 128)
    self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
    self.dec1 = self.conv_block(128, 64)
    
    # Final layer
    self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

  def conv_block(self, in_channels, out_channels):
    return nn.Sequential(
      nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
      nn.BatchNorm2d(out_channels),
      nn.LeakyReLU(),
      nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
      nn.BatchNorm2d(out_channels),
      nn.LeakyReLU()
    )

  def forward(self, x):

    # Downsampling
    enc1 = self.enc1(x)
    enc2 = self.enc2(F.max_pool2d(enc1, kernel_size=2, stride=2))
    enc3 = self.enc3(F.max_pool2d(enc2, kernel_size=2, stride=2))
    enc4 = self.enc4(F.max_pool2d(enc3, kernel_size=2, stride=2))
    
    # Bottleneck
    bottleneck = self.bottleneck(F.max_pool2d(enc4, kernel_size=2, stride=2))
    
    # Upsampling
    dec4 = self.upconv4(bottleneck)
    dec4 = torch.cat((dec4, enc4), dim=1)
    dec4 = self.dec4(dec4)
    dec3 = self.upconv3(dec4)
    dec3 = torch.cat((dec3, enc3), dim=1)
    dec3 = self.dec3(dec3)
    dec2 = self.upconv2(dec3)
    dec2 = torch.cat((dec2, enc2), dim=1)
    dec2 = self.dec2(dec2)
    dec1 = self.upconv1(dec2)
    dec1 = torch.cat((dec1, enc1), dim=1)
    dec1 = self.dec1(dec1)
    
    # Final layer
    return self.final_conv(dec1)
  
"""