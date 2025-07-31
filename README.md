# MNIST-CNN with PyTorch

**Purpose**  
PyTorch 기반으로 구현된 간단한 MNIST 숫자 분류 CNN 모델.
train.py를 통해 학습을 수행하고, eval.py를 통해 학습된 모델의 성능을 평가.
model.py, data_loader.py는 각각 모델의 정의와 데이터 로더를 구성. 

**INSTALATION**
pip install -r requirements.txt

**train**
python train.py

**evaluate**
python eval.py

**환경**
Python 3.8 이상
PyTorch 2.0 이상
torchvision 0.15 이상 

**참고사항**
학습된 모델은 mnist_cnn.pth로 저장
eval.py에서는 저장된 모델을 불러와 테스트셋의 예측 결과를 출력 
