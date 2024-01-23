folder='./models'
#'tmp/kitti/para/test0/models'
path='../datasets/20231204/testing'

CUDA_VISIBLE_DEVICES=0 python evaluate_nyu.py --load_weights_folder $folder --eval_mono --data_path $path --save_pred_disps --eval_split 1204
