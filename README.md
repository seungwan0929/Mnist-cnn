# MNIST-CNN with PyTorch

**Purpose**  

PyTorch 기반으로 구현된 간단한 MNIST 숫자 분류 CNN 모델.

train.py를 통해 학습을 수행하고, eval.py를 통해 학습된 모델의 성능을 평가.

model.py, data_loader.py는 각각 모델의 정의와 데이터 로더를 구성. 


**기본 환경 CUDA:11.8 / Windows 11**

1. conda create -n torch_env python=3.10 -y

2. conda activate torch_env

3. conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

4. conda install matplotlib pandas numpy

5. pip install -r requirements.txt


**train**

python train.py

**evaluate**

python eval.py


**참고사항**

학습된 모델은 weights/mnist_cnn.pth로 저장.

eval.py에서는 저장된 모델을 불러와 테스트셋의 예측 결과를 출력.
