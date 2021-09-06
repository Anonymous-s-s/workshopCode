CUDA_VISIBLE_DEVICES=0 python resprune.py \
--dataset cifar10 \
--arch resnet18_cifar \
--test-batch-size 128 \
--depth 18 \
--percent 0 \
--model ./poisonresult_2/1/EB-30-11.pth.tar \
--save ./poisonresult_2/1_pruned/EB-0-11 \