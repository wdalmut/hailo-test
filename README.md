# Hailo Test

## Python 

```
(.venv) ✔ ~/git/test-hailo [master L|…11] 
13:38 $ python -V
Python 3.12.3
```

## Training

```sh
python train.py --epochs 20 --batch_size 32 --learning_rate 1e-3
```

## Generate HEF

```
hailo compiler model_h8l.har --output-dir hef --hw-arch hailo8l
```
