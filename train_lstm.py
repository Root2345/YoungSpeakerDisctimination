import os, datetime, csv
import numpy as np

from tensorflow.python.ops.gen_linalg_ops import batch_cholesky
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.metrics import Precision, Recall, CategoricalAccuracy, AUC
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from .models.lstm_model import lstm_model
import my_data_loader
import random

random.seed(1234)


def set_tf_dataset(spectrograms, labels, classes, shuffle=True, batch_size=16):
    """
    tensorflowで処理可能な形式に変換
    """
    spectrograms = np.array(spectrograms)
    ds = tf.data.Dataset.from_tensor_slices((spectrograms, tf.one_hot(labels, classes)))
    if shuffle:
        ds = ds.shuffle(buffer_size=1200)
    ds = ds.batch(batch_size)
    return ds


def model_train(model, tr_ds, va_ds, ts_ds, epochs, log):
    tb = TensorBoard(log_dir=os.path.join('./logs', log),
                    histogram_freq=1,
                    write_images=True,
                    update_freq='batch',
                    )

    es = tf.keras.callbacks.EarlyStopping(verbose=1, patience=5)

    histroy = model.fit(
        tr_ds,
        epochs=epochs,
        validation_data=va_ds,
        callbacks=[tb, es],
        verbose=2
    )

    eva = model.evaluate(ts_ds)
    predictions = model.predict(ts_ds)


def valid_log(log_root, filename, write_data):
    """
    logファイルに記述
    """
    with open(os.path.join(log_root, filename), mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(write_data)


def get_evaluate(model, ds, age_th, test_group):
    """
    評価指標の取得とf値の計算
    """
    # データセットを入力してモデルを評価
    eva = model.evaluate(ds, verbose=0)
    precision = eva[2]
    recall = eva[3]
    # F値を計算して配列の最後に追加
    try:
        eva.append(2 * precision * recall / (precision + recall))
    except ZeroDivisionError:
        eva.append(0)
    

    head = ["ageth: " + str(age_th), 'group: ' + str(test_group)]
    head.extend(eva)
    print("loss: {} - categorical_accuracy: {} - precision: {} - recall: {} - auc: {} - fmeasure: {}".format(eva[0], eva[1], eva[2], eva[3], eva[4], eva[5]))
    return head


def rand_indexs_nodup(from_num, to_num):
    """
    重複しない自然数を指定した範囲・個数だけ生成
    """
    ns = []
    while len(ns) < to_num:
        n = random.randint(0, from_num-1)
        if not n in ns:
            ns.append(n)
    return ns


def choice_datas(feature, labels, num=50):
    """
    対応した特徴量とラベルを指定した個数にランダム抽出
    feature: 特徴量
    labels: ラベル
    num: 絞り込む個数(train:400, valid/test:50)
    """
    labelby_feat = []
    use_labels = []

    for i in range(len(set(labels))):
        fe = np.array(feature)[labels==i]
        rand_index = rand_indexs_nodup(from_num=len(fe), to_num=num)
        
        labelby_feat.append(fe[rand_index])
        use_labels.append(np.full(num, i))
        print(fe[rand_index].shape)

    use_feature = np.concatenate([labelby_feat[0], labelby_feat[1], labelby_feat[2]])
    use_labels = np.concatenate([use_labels[0], use_labels[1], use_labels[2]])

    return use_feature, use_labels


def conc_spec_label(specs, labels, train=1):
    conc_specs = []
    conc_labels = []

    if train==1:
        for i, cp_spec in enumerate(specs):
            for spec in cp_spec:
                conc_specs.append(spec)
                conc_labels.append(labels[i])
    else:
        for i, cp_spec in enumerate(specs):
            conc_specs.append(cp_spec[0])
            conc_labels.append(labels[i])

    return np.array(conc_specs), np.array(conc_labels)


def valid_log(log_root, filename, write_data):
    """
    logファイルに記述
    """
    with open(os.path.join(log_root, filename), mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(write_data)


def cross_valid(spectrograms, datas, classes, epochs, batch_size, learning_rate):
    print("params - classes: {}, epoch: {}, learing rate: {}".format(classes, epochs, learning_rate))
    log_root = os.path.join('/home/s226059/workspace/git_space/workspace/logs/lstm_re', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    
    test_result = []
    # test_ageth = [10]
    # test_tgroup = [2]
    for age_th in range(9, 19):
    # for age_th in test_ageth:
        
        csv_header = ["age_th: " + str(age_th), "test group", "Loss", "Accuracy", "Precision", "Recall", "AUC", "F1Score"]
        
        tr_eva_sum = []
        va_eva_sum = []
        ts_eva_sum = []
        test_result = []

        for test_group in range(0, 10):
        # for test_group in test_tgroup:

            print("年齢閾値: {}, テストグループ: {}".format(age_th, test_group))
            log = os.path.join(log_root, 'ageth' + str(age_th) + '/' + 'test' + str(test_group))
            os.makedirs(log, exist_ok=True)
            train_specs, train_labels, valid_specs, valid_labels, test_specs, test_labels = my_data_loader.load_datas(spectrograms, datas, age_th, test_group, classes)
            
            choice = True
            if choice == True:
                train_specs, train_labels = choice_datas(train_specs, train_labels, 400)
                valid_specs, valid_labels = choice_datas(valid_specs, valid_labels, 50)
                test_specs, test_labels = choice_datas(test_specs, test_labels, 50)

                train_specs, train_labels = conc_spec_label(train_specs, train_labels, train=1)
                valid_specs, valid_labels = conc_spec_label(valid_specs, valid_labels, train=0)
                test_specs, test_labels = conc_spec_label(test_specs, test_labels, train=0)

            tr_ds = set_tf_dataset(train_specs, train_labels, classes=classes, batch_size=batch_size)
            va_ds = set_tf_dataset(valid_specs, valid_labels, classes=classes, shuffle=False, batch_size=1)
            ts_ds = set_tf_dataset(test_specs, test_labels, classes=classes, shuffle=False, batch_size=1)

            model = lstm_model(classes,learning_rate)
            tb = TensorBoard(log_dir=log,
                              histogram_freq=1,
                              write_images=True,
                              update_freq='batch',
                              )

            es = tf.keras.callbacks.EarlyStopping(verbose=1, patience=10)

            history = model.fit(
                tr_ds,
                epochs=epochs,
                validation_data=va_ds,
                callbacks=[tb],
                verbose=2
            )
            # ログの記述
            if test_group == 0:
                valid_log(log_root, "train_val.csv", csv_header)
                valid_log(log_root, "valid_val.csv", csv_header)
                valid_log(log_root, "test_val.csv", csv_header)

            print("train data evaluate")
            tr_eva = get_evaluate(model, tr_ds, age_th, test_group)
            print("validation data evaluate")
            va_eva = get_evaluate(model, va_ds, age_th, test_group)
            print("test data evaluate")
            ts_eva = get_evaluate(model, ts_ds, age_th, test_group)
            valid_log(log_root, "train_val.csv", tr_eva)
            valid_log(log_root, "valid_val.csv", va_eva)
            valid_log(log_root, "test_val.csv", ts_eva)
            tr_eva_sum.append(tr_eva[2:len(tr_eva)])
            va_eva_sum.append(va_eva[2:len(va_eva)])
            ts_eva_sum.append(ts_eva[2:len(ts_eva)])

        age_results = [tr_eva_sum, va_eva_sum, ts_eva_sum]
        stages = ["train", "valid", "test"]

        for result, stage in zip(age_results, stages):
            head = ["ageth: " + str(age_th), 'group: average']
            ave = np.mean(result, axis=0)
            head.extend(ave)
            valid_log(log_root, stage + "_val.csv", head)

            if stage == "test":
                test_result.append(head)

    for r in test_result:
        valid_log(log_root, "test_val.csv", r)


def main():
    classes = 3
    epochs = 200
    batch_size = 8
    learning_rate = 1e-6
    spectrograms, datas = my_data_loader.set_spectrograms_and_datas()
    # spectrograms = spectrograms.transpose(0, 2, 1)
    # spectrograms = spectrograms.reshape(spectrograms.shape[0], 251, 257, 1)
    cross_valid(spectrograms, datas, classes, epochs, batch_size, learning_rate)


if __name__ == '__main__':
    main()
