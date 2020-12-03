# coding: utf-8
import os.path
import gzip
import pickle
import os
import numpy as np
try:
    import urllib.request
except ImportError:
    raise ImportError('You should use Python 3.x')

# 다운로드 파일(60,000개의 트레이닝 이미지 / 10,000개의 테스트 이미지)
url_base = 'http://yann.lecun.com/exdb/mnist/'
key_file = {
    'train_img': 'train-images-idx3-ubyte.gz',      # training set images
    'train_label': 'train-labels-idx1-ubyte.gz',    # training set labels
    'test_img': 't10k-images-idx3-ubyte.gz',        # test set images
    'test_label': 't10k-labels-idx1-ubyte.gz'       # test set labels
}

dataset_dir = os.path.join(os.getcwd(), 'dataset')
save_file = dataset_dir + "/mnist-neural-network.pkl"

train_num = 60000
test_num = 10000
img_dim = (1, 28, 28)
img_size = 784


def _load_label(file_name):
    file_path = dataset_dir + "/" + file_name
    
    print("Converting " + file_name + " to NumPy Array ...")
    with gzip.open(file_path, 'rb') as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8)
    print("Done")
    
    return labels


def _load_img(file_name):
    file_path = dataset_dir + "/" + file_name
    
    print("Converting " + file_name + " to NumPy Array ...")    
    with gzip.open(file_path, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, img_size)
    print("Done")
    
    return data


def _change_one_hot_label(x):
    t = np.zeros((x.size, 10))
    for idx, row in enumerate(t):
        row[x[idx]] = 1

    return t


def init_network():
    datasetdir = os.path.join(os.getcwd(), 'dataset')
    with open(datasetdir + '/sample_weight.pkl', 'rb') as f:
        return pickle.load(f)


def load_mnist(normalize=True, flatten=True, one_hot_label=False):

    """
    07.mnist-neural-network 데이터셋 읽기
    
    Parameters
    ----------
    normalize : 이미지의 픽셀 값을 0.0~1.0 사이의 값으로 정규화할지 정한다.
    one_hot_label : 
        one_hot_label이 True면、레이블을 원-핫(one-hot) 배열로 돌려준다.
        one-hot 배열은 예를 들어 [0,0,1,0,0,0,0,0,0,0]처럼 한 원소만 1인 배열이다.
    flatten : 입력 이미지를 1차원 배열로 만들지를 정한다. 
    
    Returns
    -------
    (훈련 이미지, 훈련 레이블), (시험 이미지, 시험 레이블)

    """

    # 1. dataset 초기화 작업 #########################################
    if not os.path.exists(save_file):
        # 1-1. download 07.mnist-neural-network images + labels
        for fname in key_file.values():
            file_path = dataset_dir + "/" + fname

            if os.path.exists(file_path):
                continue

            print(f'Downloading {fname} ...')
            urllib.request.urlretrieve(url_base + fname, file_path)
            print("Done")

        # 1-2. convert to numpy
        dataset = {
            'train_img': _load_img(key_file['train_img']),
            'train_label': _load_label(key_file['train_label']),
            'test_img': _load_img(key_file['test_img']),
            'test_label': _load_label(key_file['test_label'])
        }

        # 1-3. save pikle (serialize)
        print("Creating pickle file ...")
        with open(save_file, 'wb') as f:
            pickle.dump(dataset, f, -1)
        print("Done!")

    # 2. 직렬화된 dataset 로딩 ########################################
    with open(save_file, 'rb') as f:
        dataset = pickle.load(f)

    # 3. 데이터 처리(전처리) ###########################################
    # 3-1. 이미지의 픽셀값(0~255)을 0~1사이 값으로 정규화 한다.
    if normalize:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0

    # 3-2. one_hot 레벨로 변경함
    if one_hot_label:
        dataset['train_label'] = _change_one_hot_label(dataset['train_label'])
        dataset['test_label'] = _change_one_hot_label(dataset['test_label'])    

    # 3-3. flatten 되어있는 1차원(784개)배열을 1*28*28 3차원 배열로 바꿈
    if not flatten:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28)

    return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label']) 
