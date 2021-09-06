CUDA_VISIBLE_DEVICES=0 python main_add_p.py \
--dataset cifar100 \
--arch resnet18_cifar \
--depth 18 \
--lr 0.1 \
--epochs 90 \
--schedule 30 60 \
--batch-size 256 \
--test-batch-size 64 \
--save ./poisonresult_2/6 \
--momentum 0.9 \
--sparsity-regularization \
--poison_pre 2