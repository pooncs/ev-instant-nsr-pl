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
python launch.py    --config  configs/neus-evdata.yaml \
                    --gpu 0 \
                    --train \
                    model.radius=1 \
                    dataset.scene=CTTBLO004_ROI_1_0_1_masked \
                    trainer.max_steps=20000 \
                    tag=minz0_maxz1

# python launch.py    --config /home/ubuntu/src/data/SCZRIC001_ROI_4_0_1_masked/exp/ROI4@20230404-201943/config/parsed.yaml \
#                     --gpu 0 \
#                     --test \
#                     --resume /home/ubuntu/src/data/SCZRIC001_ROI_4_0_1_masked/exp/ROI4@20230404-201943/ckpt/epoch=0-step=20000.ckpt
