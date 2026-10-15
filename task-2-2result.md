Dog-Cat cnn result
================================================================
原始数据集目录 : D:\Code\Jotang-ml-task2\archive
小数据集输出到 : D:\Code\Jotang-ml-task2\small-data
每类抽取       : 500 张   （验证集占 20%）
================================================================

[cats]
  来源目录 : archive\training_set\training_set\cats
  原图总数 : 4000 张
  抽出来的 : 500 张  ->  train 400 张 + val 100 张
  写到     : train/cats  和  val/cats

[dogs]
  来源目录 : archive\training_set\training_set\dogs
  原图总数 : 4005 张
  抽出来的 : 500 张  ->  train 400 张 + val 100 张
  写到     : train/dogs  和  val/dogs

================================================================
小数据集统计
----------------------------------------------------------------
  cats  : 原图 4000 张 -> train 400 张, val 100 张  (小计 500 张)
  dogs  : 原图 4005 张 -> train 400 张, val 100 张  (小计 500 张)
  合计    : train 800 张, val 200 张  (总计 1000 张)
================================================================

复查输出目录（直接数文件个数）：
  train/cats: 400 张
  train/dogs: 400 张
  val/cats: 100 张
  val/dogs: 100 张

完成！小数据集在: D:\Code\Jotang-ml-task2\small-data
下一步：直接运行  python task-2-2.py  （它用 ImageFolder 读 small-data/train 和 small-data/val）
Using device: cuda
GPU 型号: NVIDIA GeForce RTX 5060 Laptop GPU
PyTorch 版本: 2.11.0+cu128  CUDA 版本: 12.8

本次配置: IMG_SIZE=224, BATCH_SIZE=64, EPOCHS=15, NUM_WORKERS=0
类别名: ['cats', 'dogs']   对应编号: {'cats': 0, 'dogs': 1}
训练集 800 张，验证集 200 张
Batch shape: torch.Size([64, 3, 224, 224]), Labels shape: torch.Size([64])
SimpleCNN(
  (conv1): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (bn1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (pool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv3): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (bn3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (pool3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv4): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (bn4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (pool4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (gap): AdaptiveAvgPool2d(output_size=1)
  (dropout): Dropout(p=0.5, inplace=False)
  (fc): Linear(in_features=256, out_features=2, bias=True)
)

--- Layer Shapes ---
Input:        (1, 3, 224, 224)
conv1+bn+relu: (1, 32, 224, 224)
pool1:         (1, 32, 112, 112)
conv2+bn+relu: (1, 64, 112, 112)
pool2:         (1, 64, 56, 56)
conv3+bn+relu: (1, 128, 56, 56)
pool3:         (1, 128, 28, 28)
conv4+bn+relu: (1, 256, 28, 28)
pool4:         (1, 256, 14, 14)
GAP:           (1, 256, 1, 1)
Flatten:       (1, 256)
Epoch 01/15 | Train Loss: 0.7230 Acc: 0.5687 | Val Loss: 0.7167 Acc: 0.5000
  -> Best model saved (Val Acc: 0.5000)
Epoch 02/15 | Train Loss: 0.6814 Acc: 0.5887 | Val Loss: 0.6567 Acc: 0.6150
  -> Best model saved (Val Acc: 0.6150)
Epoch 03/15 | Train Loss: 0.7006 Acc: 0.5763 | Val Loss: 0.6102 Acc: 0.6700
  -> Best model saved (Val Acc: 0.6700)
Epoch 04/15 | Train Loss: 0.6666 Acc: 0.5900 | Val Loss: 0.5954 Acc: 0.6900
  -> Best model saved (Val Acc: 0.6900)
Epoch 05/15 | Train Loss: 0.6601 Acc: 0.6188 | Val Loss: 0.6189 Acc: 0.6750
Epoch 06/15 | Train Loss: 0.6489 Acc: 0.6175 | Val Loss: 0.6014 Acc: 0.6700
Epoch 07/15 | Train Loss: 0.6567 Acc: 0.6300 | Val Loss: 0.6539 Acc: 0.5450
Epoch 08/15 | Train Loss: 0.6442 Acc: 0.6362 | Val Loss: 0.5292 Acc: 0.7400
  -> Best model saved (Val Acc: 0.7400)
Epoch 09/15 | Train Loss: 0.6514 Acc: 0.6150 | Val Loss: 0.5564 Acc: 0.7050
Epoch 10/15 | Train Loss: 0.6254 Acc: 0.6600 | Val Loss: 0.5885 Acc: 0.6550
Epoch 11/15 | Train Loss: 0.6148 Acc: 0.6613 | Val Loss: 0.5437 Acc: 0.7150
Epoch 12/15 | Train Loss: 0.6418 Acc: 0.6288 | Val Loss: 0.5989 Acc: 0.6950
Epoch 13/15 | Train Loss: 0.6252 Acc: 0.6538 | Val Loss: 0.5349 Acc: 0.7650
  -> Best model saved (Val Acc: 0.7650)
Epoch 14/15 | Train Loss: 0.6177 Acc: 0.6825 | Val Loss: 0.5461 Acc: 0.7000
Epoch 15/15 | Train Loss: 0.5997 Acc: 0.6763 | Val Loss: 0.6459 Acc: 0.6700

[最终评估] Val Loss: 0.5349, Val Acc: 0.7650
拿来做特征图可视化的图片: small-data/val\cats\cat.1017.jpg，标签 = cats