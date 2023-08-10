To Run model for window-based predictor with Integrated All Action prediction:

```
python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda  --model_training=combined --loss_type=ce_over --window_eval=true --window_size=7

```
--- 

To Run model for window-based predictor with Integrated All Action prediction + Teacher Forcing Strategy with Sampled Softmax Uniform Loss:

```
python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --loss_type=sampled_softmax --model_training=combined --window_eval=true --uniform_ss=true --strategy=teacher_forcing


```
--- 

To Run model without window:

```
python main.py --device=cuda --dataset=ml-1m --train_dir=default --inference_only=false --maxlen=200

```

---

Modified based on [paper author's tensorflow implementation](https://github.com/kang205/SASRec), switching to PyTorch(v1.6) for simplicity, executable by:

```python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda```

Check paper author's [repo](https://github.com/kang205/SASRec) for detailed intro and more complete README, and here's paper bib FYI :)

```
@inproceedings{kang2018self,
  title={Self-attentive sequential recommendation},
  author={Kang, Wang-Cheng and McAuley, Julian},
  booktitle={2018 IEEE International Conference on Data Mining (ICDM)},
  pages={197--206},
  year={2018},
  organization={IEEE}
}
```
Model base code (before all changes), is adopted from:
https://github.com/pmixer/SASRec.pytorch
