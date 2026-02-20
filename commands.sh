
python trainer_deterministic.py \
--max_epochs 3 \
--batch_size 32 \
--backbone resnet50 \
--contrastive_approach simclr \
--optimizer sgd \
--img_dir_path "/home/rkonan/Datasets/test/small_toy/images/" \
--csv_file "/home/rkonan/Datasets/test/small_toy/train.csv" \
--json_file "/home/rkonan/Datasets/test/small_toy/label_num_to_disease_map.json"





python trainer_deterministic.py \
--max_epochs 3 \
--batch_size 32 \
--backbone vit_b_16 \
--learning_rate 1e-05 \
--contrastive_approach supcon \
--optimizer adam \
--img_dir_path "/home/rkonan/Datasets/cassava-leaf-disease-classification-original/train_images/" \
--csv_file "/home/rkonan/Datasets/cassava-leaf-disease-classification-original/train.csv" \
--json_file "/home/rkonan/Datasets/cassava-leaf-disease-classification-original/label_num_to_disease_map.json"


