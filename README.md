# Take_over
田中聡久研2021年卒，庄司拓句（shoji18）の引継ぎ用リポジトリです．
AWS上で実行することを前提としているので注意してください．
※公開版では config.py など，一部のデータが削除済みです

## リポジトリの構成
- Models/:  
  実装したモデル
- Requirements/:  
  各コードを実行するために必要な環境
- data_preparation.py
- predict.py
- train_*.py
- config.py
- losses.py:  
  重み付き損失関数（Focal loss， Class-balanced focal loss）．
- my_utility.py:  
  結果保存用ディレクトリ作成，データの整形，結果の評価など，どのモデルの学習にも共通する処理を記載．
- compare_models*.py:  
  `eval_*.py` に相当．モデルごとの結果を比較するためのコード．
  
## 解析の始め方
0. anacondaが入っていて，`conda` のPATHが通っている前提です．
1. このリポジトリをクローンする．
2. 後に解説する Requirements の部分を参考に，condaの環境構築
3. `conda activate preparation` を実行
4. `python data_preparation.py` を実行
5. `conda activate meegnet` を実行
6. `srun -p gpu -N 1 --exclusive bash demo.sh` を実行
   

## 各ファイルの解説

### Models
#### model_mEEGNet.py
修論，SIP2021，EMBC2021での提案モデル．  
[Googleドライブのリンク]()

EEGNet [V. J. Lawhern et al., Journal of Neural Engineering, 2018] を元に作成．

#### model_ScalpNet.py
ICASSP2020での提案モデル．  
[Googleドライブのリンク]()

#### model_Hossain.py
Hossainらが提案した発作検出用のCNN（比較用に一部改変）．   
[Googleドライブのリンク]()   
[公式](https://dl.acm.org/doi/fullHtml/10.1145/3241056)

#### model_Zhou.py
Zhouらの発作検出，予測用モデル（比較用に一部改変）．  
[Googleドライブのリンク]()  
[公式](https://www.frontiersin.org/articles/10.3389/fninf.2018.00095/full)

#### model_SVM.py
入力した特徴は周波数領域（FFT→出力の絶対値）．  
パワースペクトルを使ってもいいと思います．
cuMLを使っているので非常に高速です．

#### model_Acharya.py
Acharyaらの発作検出，予測用モデル．  
層数が多く，時間方向の入力サイズが500（1s）程度だと，プーリングによってサイズが1になってしまいます．  
mEEGNetやその他のモデルと比較ができないので，比較対象として使用していません．  
[Googleドライブのリンク]()  
[公式](https://www.sciencedirect.com/science/article/abs/pii/S0010482517303153)


#### EEGModels.py
EEGNetのコード．著者が提供．参考用に．  
[GitHub](https://github.com/vlawhern/arl-eegmodels)


### Requirements

#### 環境一覧
- compare_models: compare_models*.py 実行時に必要な環境
- meegnet: mEEGNetの訓練に必要な環境
- preprocessing: data_preparation.py 実行時に必要な環境
- scalpnet: scalpnet の訓練に必要な環境．Hossain，Zhouもこの環境で．
- svm: svm 実行時に必要な環境

#### 導入方法
anacondaが入っていて，`conda` のPATHが通っている前提です．  
このリポジトリをクローンした後，ターミナル上で以下を実行．

```
cd Take_over/Requirements/
conda env create -f = <ファイル名（*.yml）>
```

その後，
```
conda activate <環境名>
```
で構築した環境に入れます．

### losses.py
Focal loss，Class-balanced focal loss が入っています．  
import すれば，Kerasのモデルに簡単に導入できます．
- Focal loss  
  [Googleドライブ]()  
  [公式](https://arxiv.org/abs/1708.02002)
- Class-balanced loss  
  [Googleドライブ]()  
  [公式](https://openaccess.thecvf.com/content_CVPR_2019/papers/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.pdf) 

### my_utility.py
- `mk_resultdir_*()`  
  結果を保存するディレクトリを作成する関数
- `generate_dataset_*()`  
  data_preparationで生成したnumpy配列を，モデルに入力するサンプルに分割する関数．  
  サンプルや生成の仕方はconfigで設定可能．
- `send_notification()`  
  slackのwebhookを使用すれば，学習が終わった時などにslackに通知が来ます．便利．
  [参考]()
- `eval()`  
  モデルの検出結果を評価する関数．
- `annotation_generator()`  
  結果を TJNoter で確認するための関数．区間アノテーションを出力．
- `peakfile_generator()`  
  結果を TJNoter で確認するための関数．点アノテーションを出力．

### compare_models_*.py
モデルの検出結果を比較するためのコード．  
他の人の解析にそのまま使えるかどうかは微妙ですが，  
統計的検定やLatex用の表生成のコードなどもあるので，参考までに．
#### 使い方
1. このコードを `Results/` 直下に置く
2. 最低3つのモデルで検出を実施（統計的検定にFriedman検定を使用しているため）
3. 結果が入ったディレクトリの名前を変更（任意）  
  例:`mv train_Model1 Model1`
4. 症例間検証の場合，  
`python3 compare_models_intersub "Model1" "Model2" "Model3"` を実行．  
交差検証の場合，
`python3 compare_models_cv "Model1" "Model2" "Model3"` を実行．
