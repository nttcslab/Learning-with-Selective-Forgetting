# Learning with Selective Forgetting (IJCAI21)
This is the official PyTorch impelementation of our paper "Learning with Selective Forgetting" (IJCAI21).

![cover](imgs/cover.png )

## Generate dataset and mnemonic code (CIFAR100 with Task2, Task5, Task10)
`sh ./gen_datasets.sh`

## Run main process (CIFAR100)
`sh ./cifarT2.sh` # CIFAR100 with Task2

`sh ./cifarT5.sh` # CIFAR100 with Task5

`sh ./cifarT10.sh` # CIFAR100 with Task10

## Citation
If you use this toolbox or benchmark in your research, please cite this project.

```
@inproceedings{shibata2021learning,
  title={Learning with Selective Forgetting.},
  author={Shibata, Takashi and Irie, Go and Ikami, Daiki and Mitsuzumi, Yu},
  booktitle={IJCAI},
  volume={2},
  number={4},
  pages={6},
  year={2021}
}
```