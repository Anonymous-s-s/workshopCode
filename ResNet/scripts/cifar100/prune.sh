CUDA_VISIBLE_DEVICES=0 python resprune.py \
--dataset cifar100 \
--arch resnet18_cifar \
--test-batch-size 128 \
--depth 18 \
--percent 0.7 \
--model ./poisonresult_2/4/EB-70-17.pth.tar \
--save ./poisonresult_2/4_pruned/EB-70-17 \