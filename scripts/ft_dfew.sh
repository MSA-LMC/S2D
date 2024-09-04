# pip install tensorboardX scikit-learn  einops  timm==0.6.12 opencv-python pandas matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple

server=164
pretrain_dataset='affectnet7'
pretrain_server=3090
finetune_dataset='dfew'
num_labels=7
ckpts=(ckpts/Pretrain-AffectNet-7.pth)
input_size=224
sr=1
model="affectnet_pretrain"
model_dir="${model}_server${pretrain_server}"


lr=1e-5
epochs=100

BATCH_SIZE=8
device=0,1 

splits=(1 2 3 4 5)
for ckpt in "${ckpts[@]}";
do
for split in "${splits[@]}";
do
    tag="${nf}#affectnet_pretrain"

    OUTPUT_DIR="./saved/model/finetuning/${finetune_dataset}/${pretrain_dataset}/${ckpt}/split0${split}_eval_lr_${lr}_epoch_${epochs}_${augment}Augment_size${input_size}_sr${sr}#tag${tag}"
    if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p $OUTPUT_DIR
    fi
    cp "$(basename "$0")" $OUTPUT_DIR/
    # path to split files (train.csv/val.csv/test.csv)
    DATA_PATH="data/DFEW/Clip/clip_224x224_16f" 
    TRAIN_LABEL="data/DFEW/EmoLabel_DataSplit/train(single-labeled)/train_set_${split}.csv"
    TEST_LABEL="data/DFEW/EmoLabel_DataSplit/test(single-labeled)/test_set_${split}.csv"
    
    MODEL_PATH="${ckpt}"
    echo $OUTPUT_DIR $MODEL_PATH
    # batch_size can be adjusted according to number of GPUs
    CUDA_VISIBLE_DEVICES=$device python  \
        run_class_finetuning.py \
        --model s2d_base_patch16_224 \
        --data_set DFEW \
        --nb_classes ${num_labels} \
        --data_path ${DATA_PATH} \
        --train_label_path ${TRAIN_LABEL} \
        --test_label_path ${TEST_LABEL} \
        --finetune ${MODEL_PATH} \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --batch_size ${BATCH_SIZE} \
        --num_sample 1 \
        --input_size ${input_size} \
        --short_side_size ${input_size} \
        --save_ckpt_freq 1000 \
        --num_frames 16 \
        --sampling_rate ${sr} \
        --opt adamw \
        --lr ${lr} \
        --opt_betas 0.9 0.999 \
        --weight_decay 0.05 \
        --epochs ${epochs} \
        --dist_eval \
        --test_num_segment 2 \
        --test_num_crop 2 \
        --num_workers 8 \
        --K 2 \
        --qs 16 \
        --sdl_update_freq 40 \
        >>${OUTPUT_DIR}/nohup.out 2>&1
done
done
echo 'done!'
