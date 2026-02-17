
python trainer_deterministic.py \
--max_epochs 3 \
--batch_size 32 \
--backbone resnet50 \
--contrastive_approach simclr \
--optimizer sgd \
--img_dir_path "/home/rkonan/Datasets/test/small_toy/images/" \
--csv_file "/home/rkonan/Datasets/test/small_toy/train.csv" \
--json_file "/home/rkonan/Datasets/test/small_toy/label_num_to_disease_map.json"

