# ---------------------------------------------------------------
# 0.各ライブラリ
# ---------------------------------------------------------------
import os
# ハッシュを生成して重複判定に使用する
import hashlib
# 画像を読み込んでハッシュ計算用に整形
from PIL import Image
# Google画像検索で自動収集するモジュール
from icrawler.builtin import GoogleImageCrawler
# ファイル名を一意にするために追加
import uuid

import shutil
    

# ---------------------------------------------------------------
# 1.ハッシュの取得関数
# ---------------------------------------------------------------
def get_image_hash(image_path):
    # 画像のMD5ハッシュを作成・取得
    # →同じ画像は同じハッシュ値になる
    try:
        with Image.open(image_path) as img:
            # 画像を64×64にリサイズ、RGBに変換
            img = img.resize((64, 64)).convert("RGB")
            return hashlib.md5(img.tobytes()).hexdigest()
    except Exception:
        return None
    

# ---------------------------------------------------------------
# 2.保存済みの画像のハッシュ一覧取得
# ---------------------------------------------------------------
def get_existing_hashes(folder_path):
    # 取得したすべてのハッシュをsetに格納
    hashes = set()
    for fname in os.listdir(folder_path):
        path = os.path.join(folder_path, fname)
        if os.path.isfile(path):
            h = get_image_hash(path)
            if h:
                hashes.add(h)
    return hashes
    

# ---------------------------------------------------------------
# 3.一時フォルダからユニークな画像のみ保存先に移動
# ---------------------------------------------------------------
def save_unique_images_from_temp(temp_dir, target_dir, existing_hashes, max_total):
    new_count = 0
    # temp_dirに保存された画像をすべてチェック
    for fname in os.listdir(temp_dir):
        if len(existing_hashes) >= max_total:
            break  # 既に最大に達していれば終了
        
        fpath = os.path.join(temp_dir, fname)
        h = get_image_hash(fpath)
        # ハッシュが重複していなければ→target_dirに移動
        if h and h not in existing_hashes:
            # ファイル名の衝突を避けるためにUUIDを使ってリネーム
            ext = os.path.splitext(fname)[1]
            unique_name = f"{uuid.uuid4().hex}{ext}"
            new_path = os.path.join(target_dir, unique_name)
            os.rename(fpath, new_path)
            
            existing_hashes.add(h)
            new_count += 1
        # 重複していれば削除
        else:
            os.remove(fpath)
    # 新しく追加された画像の枚数を返す
    return new_count


# ---------------------------------------------------------------
# 4.重複画像を除外して複数キーワードから200枚画像収集
# ---------------------------------------------------------------
def crawl_images_no_duplicates(keywords, target_dir, max_total=200, max_per_keyword=50):
    # terget_dirがなければ作成
    os.makedirs(target_dir, exist_ok=True)
    # 既に保存されている画像のハッシュ一覧を取得
    existing_hashes = get_existing_hashes(target_dir)
    
    # 各キーワードに対して重複チェック用に一時フォルダを準備
    for keyword in keywords:
        if len(existing_hashes) >= max_total:
            print("✅ 収集完了：200枚に到達しました")
            break
        
        print(f"検索中: {keyword}")
        temp_dir = os.path.join(target_dir, "_temp")
        os.makedirs(temp_dir, exist_ok=True)
        
        # icrawerを使ってキーワード検索で画像をtemp_dirにダウンロード
        crawler = GoogleImageCrawler(storage={"root_dir": temp_dir})
        crawler.crawl(keyword=keyword, max_num=max_per_keyword)
        
        # ダウンロードした画像から重複なし画像を本保存
        added = save_unique_images_from_temp(temp_dir, target_dir, existing_hashes, max_total)
        print(f"{added}枚追加されました ({keyword})  現在の合計： {len(existing_hashes)}")
    
        # 最後に一時フォルダ_tempを削除
        # os.rmdir(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)

    print(f"🎉 最終的に保存された画像枚数： {len(existing_hashes)}")


# ---------------------------------------------------------------
# 5.使用
# ---------------------------------------------------------------
if __name__ == "__main__":
    keywords = ['イーブイ','イーブイ ぬいぐるみ', 'イーブイ アニメ', 'イーブイ フィギュア', 'イーブイ ゲーム', 'イーブイ イラスト']
    target_dir = r"C:\Users\harap\OneDrive\デスクトップ\pokemon-main\pokemon_dataset\img_All\133_Eevee"
    crawl_images_no_duplicates(keywords, target_dir, max_total=200, max_per_keyword=50)