# YoungSpeakerDisctimination
話者年齢埋め込み特徴を用いた若年話者判別手法の研究に使用したコードです。
これらのコードを実行するためには、VoxCeleb2, AgeVoxCeleb, 実環境発話音声データセットが必要です。

### model
モデルの構造を記述したpythonファイルを格納したディレクトリです。
- cnn_model.py
- lstm_model.py
- sinctdnn.py
- sinctdnn_age.py

### train_embedding
埋め込みモデル学習のためのpythonファイルを格納したディレクトリです。
- XvecDataloader.py
  - 埋め込み抽出モデル学習のためのデータローダを構築します。
- trainAgevector.py
  - AgeVoxCelebを用いて年齢推定タスクでSincTDNNモデルを学習します。
- trainXvector.py
  - VoxCeleb2またはAgeVoxCelebを用いて話者識別タスクでSincTDNNモデルを学習します。

### README.md
このファイルです。

### extract_vector.py
学習済みの特徴抽出モデルを用いて、音声サンプルをから埋め込みベクトルを抽出します。

### my_data_loader.py
判別モデル学習の際に使用するデータローダファイルです。

### train.py
埋め込みベクトルを入力とし、全結合NNを用いて3クラス分類を行うモデルを構築します。

### train_cnn.py
スペクトログラムを入力とし、CNNを用いて3クラス分類を行うモデルを構築します。

### train_lstm.py
スペクトログラムを入力とし、LSTMを用いて3クラス分類を行うモデルを構築します。
