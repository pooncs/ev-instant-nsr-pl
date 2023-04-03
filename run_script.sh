python launch.py    --config  configs/neus-evdata.yaml \
                    --gpu 0 \
                    --train \
                    dataset.scene=SCZRIC001_ROI_3_0_25_masked \
                    dataset.far_plane=4.0 \
                    dataset.near_plane=2.0 \
                    system.loss.lambda_rgb_mse=0. \
                    system.loss.lambda_rgb_l1=10. \
                    tag=near2_far4_mseloss0_l1_10

python launch.py    --config  configs/neus-evdata.yaml \
                    --gpu 0 \
                    --train \
                    dataset.scene=SCZRIC001_ROI_3_0_25_masked \
                    dataset.far_plane=4.0 \
                    dataset.near_plane=1.0 \
                    system.loss.lambda_rgb_mse=0. \
                    system.loss.lambda_rgb_l1=20. \
                    tag=near1_far4_mseloss0_l1_20

python launch.py    --config  configs/neus-evdata.yaml \
                    --gpu 0 \
                    --train \
                    dataset.scene=SCZRIC001_ROI_3_0_25_masked \
                    dataset.far_plane=4.0 \
                    dataset.near_plane=2.0 \
                    system.loss.lambda_rgb_mse=0. \
                    system.loss.lambda_rgb_l1=20. \
                    tag=near2_far4_mseloss0_l1_20