# 必要ライブラリのインポート

import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib

import torch
from torch import tensor
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
# from torchviz import make_dot
import torchvision.models as models
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from utils import show_images_labels
from utils import torch_seed
from utils import fit
from utils import evaluate_history

# warning表示off
import warnings
warnings.simplefilter('ignore')

# matplotlibのデフォルト表示をカスタマイズ
# デフォルトフォントサイズ変更
plt.rcParams['font.size'] = 14

# デフォルトグラフサイズ変更
plt.rcParams['figure.figsize'] = (6,6)

# デフォルトで方眼表示ON
plt.rcParams['axes.grid'] = True

# numpyの表示桁数設定
np.set_printoptions(suppress=True, precision=5)

# GPUチェック
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# transforms定義
# 訓練データ用
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5), # データ水増し(50％の確率で左右反転)
    transforms.Resize((224,224)), # 224×224リサイズ
    transforms.ToTensor(), # テンソル変換
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), # 平均0.5、標準偏差0.5で正規化
    # transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False) # 画像の一部をランダムに黒く塗りつぶす
])

# 検証データ用
val_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

# テストデータ用
test_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


# データセット定義
# classes = ['006_Charizard', '025_pikachu', '658_Greninja']
temp_dataset = ImageFolder(root='pokemon_dataset/split_dataset(img_All)/train')
classes = temp_dataset.classes

# train_dataset = ImageFolder(root='pokemon_dataset/split_dataset/train',transform=train_transform)
# val_dataset = ImageFolder(root='pokemon_dataset/split_dataset/val', transform=val_transform)
# test_dataset = ImageFolder(root='pokemon_dataset/split_dataset/test', transform=test_transform)

train_dataset = ImageFolder(root='pokemon_dataset/split_dataset(img_All)/train',transform=train_transform)
val_dataset = ImageFolder(root='pokemon_dataset/split_dataset(img_All)/val', transform=val_transform)
test_dataset = ImageFolder(root='pokemon_dataset/split_dataset(img_All)/test', transform=test_transform)

# データ数表示
print(f'訓練データ: {len(train_dataset)}件')
print(f'検証データ: {len(val_dataset)}件')
print(f'テストデータ: {len(test_dataset)}件')

# バッチサイズ・データローダー定義
batch_size = 32

# 学習データ
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# 検証データ
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
# テストデータ
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 訓練データ表示
show_images_labels(train_loader, classes, None, None)
# 検証データ表示
show_images_labels(val_loader, classes, None, None)
# テストデータ表示
show_images_labels(test_loader, classes, None, None)

# 学習済みモデルの読み込み
net = models.vgg16_bn(pretrained=True)

# すべてのパラメータの学習を無効化⇒最後の全結合層だけ
for param in net.parameters():
    param.requires_grad = False
    
# 乱数固定
torch_seed()

# クラス数を自動で取得
num_classes = len(classes) # 更新中
# VGG16の最後の全結合層の入力次元取得
in_features = net.classifier[6].in_features
# 学習される線形層
net.classifier[6] = nn.Linear(in_features, num_classes)

# AdaptiveAvgPool2dの取り外し
# net.avgpool = nn.Identity()

# GPUの利用
net = net.to(device)

# 学習率
lr = 0.001

# 損失関数
criterion = nn.CrossEntropyLoss()

# 最適化関数定義
# パラメータ修正の対象を最終ノードに限定
optimizer = optim.SGD(net.classifier[6].parameters(), lr=lr, momentum=0.9)

# historyファイルも初期化
history = np.zeros((0, 5))


# 学習の実行
num_epochs = 10
history = fit(net, optimizer, criterion, num_epochs, train_loader, val_loader, device, history)

# テストデータで最終評価を行う
net.eval()
test_loss, test_correct, n_test = 0.0, 0.0, 0

for inputs_test, labels_test in test_loader:
    inputs_test = inputs_test.to(device)
    labels_test = labels_test.to(device)
    outputs_test = net(inputs_test)
    
    loss = criterion(outputs_test, labels_test)
    test_loss += loss.item() * inputs_test.size(0)
    predicted = torch.max(outputs_test, 1)[1]
    test_correct += (predicted == labels_test).sum().item()
    n_test += labels_test.size(0)
    
avg_test_loss = test_loss / n_test
test_accuracy = test_correct / n_test


# 結果サマリー
evaluate_history(history)

# 予測結果表示
torch_seed()
show_images_labels(test_loader, classes, net, device)
# 最終結果表示
print(f'\n テストデータ：loss={avg_test_loss:.5f}, acc={test_accuracy:.5f}')

# モデルを保存
torch.save(net.state_dict(), 'model.pth')

