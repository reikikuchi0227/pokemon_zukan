# ポケモン図鑑的な機械学習モデル

## 概要
機械学習を用いたポケモンの画像分類モデルです。  
⇒※現在アプリケーションとして開発中※  
※街中等で撮影した画像を何のポケモンか教えてくれるポケモン図鑑的なアプリ

## 環境・ライブラリ
・Pytorch  
・matplotlib

## モデル内容
・pokemon_dataset：画像セット(img, img_All, img_similar, split_dataset, split_dataset(img_All), split_dataset(similar))  
・dataset.py：フォルダ(ポケモン)ごとに訓練・検証・テストデータに分割(比率:7:1.5:1.5)  
・image_gathering.py：任意のポケモンの画像を複数ワードでGoogle検索し、重複の無いよう画像を収集してキャラクターごとのデータセットを作成  
・pokemon_zukan.py：VGG16の転移学習を用いた機械学習分類モデル(モデル核部分)  
・utils.py：機械学習に必要な共通関数を定義  

## 参考資料
以下の方々・書籍等を参考に作成しました。  
・ピカチュウ判別器(https://github.com/33taro/pikachu_image_classification)  
・最短コースでわかる Pytorch&深層学習プログラミング(https://github.com/makaishi2/pytorch_book_info/blob/main/notebooks.md)
