CUDA_VISIBLE_DEVICES=1 python main_c_new_save_record.py \
--dataset cifar10 \
--arch resnet18_cifar \
--depth 18 \
--lr 0.1 \
--epochs 130 \
--schedule 70 100 \
--batch-size 256 \
--test-batch-size 64 \
--save ./poisonresult/1_retrain_record/EB-70-13 \
--momentum 0.9 \
--sparsity-regularization \
--scratch ./poisonresult/1_pruned/EB-70-13/pruned.pth.tar \
--start-epoch 32 \
--poison_pre 1 \
--poison_method 1