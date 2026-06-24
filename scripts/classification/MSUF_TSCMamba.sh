if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/classification" ]; then
    mkdir ./logs/classification
fi
if [ ! -d "./csv_results" ]; then
    mkdir ./csv_results
fi

if [ ! -d "./csv_results/classification" ]; then
    mkdir ./csv_results/classification
fi

model_name=MSUF_TSCMamba

root_path_name=./datasets/SpokenArabicDigits
model_id_name=SpokenArabicDigits
data_name=UEA

random_seed=2024

/home/zyt/miniconda3/envs/tscmamba/bin/python -u run.py \
    --task_name classification \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path $root_path_name \
    --model_id $model_id_name \
    --model $model_name \
    --data $data_name \
    --dropout 0.2 \
    --d_model 512 \
    --dconv 4 \
    --d_state 128 \
    --e_fact 2 \
    --projected_space 64 \
    --num_mambas 1 \
    --des 'Exp' \
    --lradj 'cosine' \
    --comment 'MSUF input fusion' \
    --max_pooling 1 \
    --train_epochs 200 \
    --itr 1 --batch_size 32 --learning_rate 0.0001 >logs/classification/$model_name'_'$model_id_name.log
