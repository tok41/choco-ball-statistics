# AngelClassifier

- チョコボールのパッケージ画像からエンゼルの有無を予測することが可能か否かを試行する

## Requirement
動作確認済みの環境。  
バージョンは多少違っていてもきっと動く。

- Python 3.6.4 (Anaconda)
- tensorflow-gpu 1.8.0

## 手順
### パッケージ画像の前処理
- パッケージ画像の余分な領域を削除する

```
$ cd choco-ball-statistics/analysis
$ python package-image-preprocess.py
```

### 定義ファイルの作成
- パッケージ画像のラベル付きリストを作成

```
$ python make-image-def.py
$ python make-image-def.py --help で引数確認

defaultで以下のファイルが作成される
 - image_list_test.csv
 - image_list_train.csv
```
### 学習処理
