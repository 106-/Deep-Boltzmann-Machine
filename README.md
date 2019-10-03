Deep-Boltzmann-Machine
===
(実験用)ディープボルツマンマシンの実装.

生成DBMからサンプリングしたデータを学習し, 2つのモデル間のカルバックライブラー情報量(Kullback-Leibler Divergence; KLD)を計算できます. 学習は近似学習のみ行い, 事前学習はやっていません(後に追加するかも)

## 使い方
_Python3.xが必要です._
```
$ git clone https://github.com/106-/Deep-Boltzmann-Machine.git DBM
$ cd DBM
```
サブリポジトリのファイルを持ってくる
```
$ git submodule update --init --recursive
```
必要モジュールのインストール
```
$ pip install -r ./requirements.txt
```

## 実行
```
./train_main.py (学習設定ファイル) (学習エポック)
```
その他のオプションもいろいろあります(`-h`オプションで表示)
```
$ ./train_main.py ./config/555.json 5
2019-09-19 18:28:45,027 : [INFO] Optimizer: Adamax
2019-09-19 18:28:45,027 : [INFO] Train started.
2019-09-19 18:28:45,029 : [INFO] [ 0.0 / 5 ]( 0 / 250 ) KL-Divergence: 0.8265922278966866
2019-09-19 18:28:47,301 : [INFO] [ 1.0 / 5 ]( 50 / 250 ) KL-Divergence: 0.4260760680089297
2019-09-19 18:28:48,926 : [INFO] [ 2.0 / 5 ]( 100 / 250 ) KL-Divergence: 0.25593571653305747
2019-09-19 18:28:50,547 : [INFO] [ 3.0 / 5 ]( 150 / 250 ) KL-Divergence: 0.1951527543946976
2019-09-19 18:28:52,179 : [INFO] [ 4.0 / 5 ]( 200 / 250 ) KL-Divergence: 0.16937845485757086
2019-09-19 18:28:53,804 : [INFO] [ 5.0 / 5 ]( 250 / 250 ) KL-Divergence: 0.15574382966159456
2019-09-19 18:28:53,804 : [INFO] Train ended.
2019-09-19 18:28:53,804 : [INFO] Model parameters was dumped to: ./results/2019-09-19_18-28-53_model.json
2019-09-19 18:28:53,805 : [INFO] Learning log was dumped to: ./results/2019-09-19_18-28-53_log.json
```

## 参考文献
R. Salakhutdinov and G. Hinton: [Deep Boltzmann Machines](http://proceedings.mlr.press/v5/salakhutdinov09a/salakhutdinov09a.pdf), Artificial intelligence and statistics, pp.448-455, 2009.
M. Yasuda: [Learning Algorithm of Boltzmann Machine Based on Spatial Monte Carlo Integration Method](https://www.mdpi.com/1999-4893/11/4/42/htm),  Algorithms 11(4), 2018.