from sklearn.model_selection import train_test_split
import os
import shutil
from glob import glob

# original_dir：元画像フォルダ
# output_dir：保存先フォルダ
# train_ratio=0.7, val_ratio=0.15, test_ratio=0.15：訓練：検証：テスト＝7：1.5：1.5
# 画像セットを分割する関数
def split_dataset(original_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    classes = os.listdir(original_dir) # フォルダを取得
    
    # フォルダごとに訓練、検証、テストデータに分ける
    for cls in classes:
        cls_dir = os.path.join(original_dir, cls)
        images = glob(os.path.join(cls_dir, '*')) # 画像一覧をglobで習得
        train_imgs, temp_imgs = train_test_split(images, test_size=(1- train_ratio), random_state=42) # random_stateは再現性を保つための種
        val_imgs, test_imgs = train_test_split(temp_imgs, test_size=(test_ratio / (test_ratio + val_ratio)), random_state=42)
        
        # コピー先のフォルダを作成して保存
        # train,val,testのそれぞれのクラス別フォルダを作成
        for split_name, split_imgs in zip(['train', 'val', 'test'], [train_imgs, val_imgs, test_imgs]):
            split_cls_dir = os.path.join(output_dir, split_name, cls)
            os.makedirs(split_cls_dir, exist_ok=True)
            # 対応する画像をそのフォルダにコピー
            for img_path in split_imgs:
                shutil.copy(img_path, split_cls_dir)

split_dataset('pokemon_dataset/img_All', 'pokemon_dataset/split_dataset(img_All)')