# ArcFace 찍먹


## 찍먹들

* ArcFace - https://github.com/edp1096/hello-arcface
* Faster R-CNN - https://github.com/edp1096/hello-faster-rcnn


## 실행

* 데이터셋 분리
```powershell
python3 scripts/split_dataset.py
```

* STL10
```powershell
python3 train_model.py
python3 test_model.py
python3 create_scatter.py
python3 predict.py
```


## 참고

* Datasets
    * https://pytorch.org/vision/stable/datasets.html
