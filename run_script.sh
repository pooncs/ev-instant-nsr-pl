python launch.py    --config  configs/neus-evdata.yaml \
                    --gpu 0 \
                    --train \
                    dataset.scene=SCZRIC001_ROI_3_0_25_masked \
                    dataset.far_plane=4.0 \
                    dataset.near_plane=2.0 \
                    system.loss.lambda_rgb_mse=0. \
                    system.loss.lambda_rgb_l1=20. \
                    trainer.max_steps=30000 \
                    tag=mseloss-0_l1-20_grid-256-256-100_ms30000_minz-05


