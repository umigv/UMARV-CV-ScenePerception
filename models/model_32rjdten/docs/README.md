# RNN Model using ConvLSTM
### by Parsanna Koirala

# Model description
- ConvLSTM: Processes temporal information across frames
- Encoder: Extracts hierarchical features through downsampling
- Decoder: Reconstructs high-resolution segmentation maps
- Skip Connections: Preserves fine spatial details

# Features
- Temporal-aware lane detection using ConvLSTM
- U-Net style encoder-decoder architecture
- Multi-frame processing for improved stability
- Binary segmentation of driveable areas

### Requirements
Change the `train_data, val_data = train_test_split(train_val_data, test_size=val_ratio, random_state=random.randint(1, 100))`
to `train_data, val_data = train_test_split(train_val_data, test_size=val_ratio, random_state=None, shuffle=False)` in order to make sure that
the training dataset is continuous rather than random frames.