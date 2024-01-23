# run the code on indoor set
#NYUV2
path='/home/zhonglinghui/zxm/datasets/20231204/training'
weights_folder='/home/zhonglinghui/zxm/GasMono/models'
gpus=1
#sleep 1m
CUDA_VISIBLE_DEVICES=$gpus python train.py --model_name 1204 --split 1204 --dataset nyu --height 256 --width 320 --data_path $path --learning_rate 1e-5 --use_posegt --www 0.2 --wpp 0.2 --iiters 2 --selfpp --batch_size 12 --disparity_smoothness 1e-4 --load_weights_folder $weights_folder

