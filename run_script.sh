# python launch.py    --config  configs/neus-evdata.yaml \
#                     --gpu 0 \
#                     --train \
#                     dataset.scene=SCZRIC001_ROI_3_0_25_masked \
#                     dataset.far_plane=4.0 \
#                     dataset.near_plane=2.0 \
#                     system.loss.lambda_rgb_mse=0. \
#                     system.loss.lambda_rgb_l1=20. \
#                     trainer.max_steps=30000 \
#                     tag=mseloss-0_l1-20_grid-256-256-100_ms30000_minz-05

# python launch.py    --config /home/ubuntu/src/data/SCZRIC001_ROI_4_0_1_masked/exp/ROI4@20230404-201943/config/parsed.yaml \
#                     --gpu 0 \
#                     --test \
#                     --resume /home/ubuntu/src/data/SCZRIC001_ROI_4_0_1_masked/exp/ROI4@20230404-201943/ckpt/epoch=0-step=20000.ckpt


# python launch.py    --config /home/ubuntu/src/data/SCZRIC001_ROI_3_0_25_masked/exp/test12_high@20230404-183637/config/parsed.yaml \
#                     --gpu 0 \
#                     --test \
#                     --resume /home/ubuntu/src/data/SCZRIC001_ROI_3_0_25_masked/exp/test12_high@20230404-183637/ckpt/epoch=0-step=20000.ckpt

#------------------------------------------------------------------------------------
scene="CTTBLO004_ROI_1v4_masked"
exp_name="exp1_nerf_10k"
multi_roi="F"
config="configs/nerf-evdata.yaml"
root="/home/ubuntu/src"
maxsteps=5000
gpu="0"

if [ $multi_roi == 'T' ]; then
    for i in $(seq -w 0 3); do
        for j in $(seq -w 0 3); do
            echo "Training: $scene/sub_roi_$i$j"
            python launch.py    --config $config \
                                --gpu $gpu \
                                --train \
                                model.radius=1 \
                                dataset.scene=$scene/sub_roi_$i$j \
                                trainer.max_steps=$maxsteps \
                                tag=$exp_name
            
            echo "Testing: $scene/sub_roi_$i$j"
            python launch.py    --config $root/data/$scene/exp/$exp_name*/config/parsed.yaml \
                                --gpu $gpu \
                                --test \
                                --resume $root/data/$scene/exp/$exp_name*/ckpt/epoch=0-step=$maxsteps.ckpt
        done
    done
else
    python launch.py    --config $config \
                        --gpu $gpu \
                        --train \
                        model.radius=1 \
                        dataset.scene=$scene \
                        trainer.max_steps=$maxsteps \
                        tag=$exp_name

    python launch.py    --config $root/data/$scene/exp/$exp_name*/config/parsed.yaml \
                        --gpu $gpu \
                        --test \
                        --resume $root/data/$scene/exp/$exp_name*/ckpt/epoch=0-step=$maxsteps.ckpt
fi


