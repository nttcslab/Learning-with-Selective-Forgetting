mkdir result
mkdir result/01_CIFAR_T2
mkdir result/01_CIFAR_T5
mkdir result/01_CIFAR_T10
python ./gen_data/datasets_cifar100_2.py 
python ./gen_data/datasets_cifar100_5.py 
python ./gen_data/datasets_cifar100_10.py 
python ./gen_mnemonic_cifar.py
python ./gen_mnemonic_stn.py
python ./gen_mnemonic_cub.py