# Statistics of Choco-Ball

- チョコボールの統計データ分析をします
- http://chocolate-ball.hatenablog.com/

## Required

- python : 3.7.0
- requirements.txt

## Directory

- data : DBファイルの更新などデータ関連
- analysis : データ分析用

## Setup
### Python Environment
- install Python libraries

```
$ pip install -r requirements.txt
```

### create(insert) DB
- insertするデータをCSV形式で用意

```
$ cd data
$ python insert_chocoball.py --file {csv_file_name}
```

- `choco-ball.db`という名前でsqliteのDBファイルができる

