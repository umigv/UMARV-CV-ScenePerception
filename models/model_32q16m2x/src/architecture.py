import torch.nn as nn

class scene_perception_model(nn.Module):

  # Basic CNN model for scene perception
  def __init__(self, lookback):
    super(scene_perception_model, self).__init__()
    self.model = nn.Sequential(
      nn.Conv2d( in_channels=3+3*lookback['count'], out_channels=20 , kernel_size=15 , padding=7 , stride=1 ),
      nn.BatchNorm2d(20),
      nn.LeakyReLU(),
      nn.Conv2d( in_channels=20 , out_channels=20 , kernel_size=15 , padding=7 , stride=1 ),
      nn.BatchNorm2d(20),
      nn.LeakyReLU(),
      nn.Conv2d( in_channels=20 , out_channels=20 , kernel_size=15 , padding=7 , stride=1 ),
      nn.BatchNorm2d(20),
      nn.LeakyReLU(),
      nn.Conv2d( in_channels=20 , out_channels=10 , kernel_size=15 , padding=7 , stride=1 ),
      nn.BatchNorm2d(10),
      nn.LeakyReLU(),
      nn.Conv2d( in_channels=10 , out_channels=4 , kernel_size=15 , padding=7 , stride=1 ),
    )

  def forward(self, input):
    output = self.model(input)
    return output