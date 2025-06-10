python train.py --data texas --lr 0.01 --hidden 128 --dropout 0.7 --alpha 0.9 --k 3 --delta 0.9 --beta 0.01 --estimator "mlp" --large_scale 0 --eps 0.01 --r 0.3 --patience 200 --lamda 0.5 --weight 0.7
python train.py --data wisconsin  --lr 0.005 --hidden 512 --dropout 0.1 --alpha 0.8 --k 5 --delta 0.4 --beta 0.5 --estimator "mlp" --large_scale 0 --eps 0.005 --r 0.5 --patience 200  --lamda 0.3 --weight 0.8
python train.py --data cornell --lr 0.01 --hidden 64 --dropout 0.2 --alpha 0.9 --k 10 --delta 0.3 --beta 0.01 --estimator "mlp" --large_scale 0 --eps 0.01 --r 0.3 --patience 200 --lamda 0.7 --weight 0.7
python train.py --data squirrel --lr 0.01 --hidden 512 --dropout 0.0 --alpha 0.9 --k 3 --delta 15 --beta 0.01 --eps 0.001 --estimator "gcn" --large_scale 1 --r 0.7 --patience 100 --lamda 0.9 --weight 0.1
python train.py --data chameleon --lr 0.01 --hidden 512 --dropout 0.1 --alpha 0.9 --k 3 --delta 20 --beta 0.01 --estimator "gcn" --eps 0.001  --large_scale 1 --r 0.7 --patience 100 --lamda 0.9 --weight 0.7
python train.py --data actor --lr 0.005 --weight_decay 0.005 --hidden 128 --dropout 0.1 --alpha 0.5 --k 10 --delta 0.3 --beta 0.01 --estimator "mlp" --large_scale 1 --r 0.3 --eps 0.001 --patience 40 --lamda 0.1 --weight 0.4
python train.py --data cora --lr 0.01 --hidden 64 --dropout 0.7 --alpha 0.8 --k 32 --delta -0.5 --beta 0.01 --estimator "gcn" --large_scale 0 --eps 0.001 --lamda 0.2 --patience 40 --weight 0.6
python train.py --data pubmed --lr 0.01 --hidden 128 --dropout 0  --alpha 0.8 --k 6 --delta -0.3 --beta 0.01 --estimator "gcn" --large_scale 1 --eps 0.001 --lamda 0.2 --patience 40 --weight 0.4

