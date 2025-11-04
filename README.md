# PhysIRL
##### Training
For example to train on GTA-IM dataset with 1 second input(5 frames), 2 seconds output (5 frames) with 5/11 (5 sgcnn, 11 txcnn) layers execute this command. 
```bash
python train.py --lr 0.01 --n_stgcnn 5 --n_txpcnn 11  --dataset GTA_IM --use_lrschd --num_epochs 450  --tag 1 
```
