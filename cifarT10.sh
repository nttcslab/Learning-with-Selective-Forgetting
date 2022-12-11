# sh ./cifarT10.sh > /dev/null 2>&1 &
export CUDA_VISIBLE_DEVICES=0,1,2,3
fini=./ini/cifar_t10-r3-E1.ini
fdir=01_CIFAR_T10
main=./main_CIFAR_T10.py
fkey=./cifar_Rand4_n10.pt

pdir=./result/${fdir}
mkdir $pdir

ewc=100
ewc_mas=5
lmdlwf=5
lmdkeep=10
lmdcnt=0

mode=MH_MT
epch=200
lrsch=40,60,120,160,200

odir=${fdir}/10_Prop_EWC+_LwF+_LF${lmdlwf}_K${lmdkeep}_EW${ewc}_C${lmdcnt}
python $main     --lr-sch $lrsch --ini-file $fini --mode $mode  --keys-dataset $fkey  --outdir $odir  --epochs-per-task $epch --lmd-ewc $ewc   --lmd-lwf $lmdlwf --lmd-keep $lmdkeep --lmd-cnt $lmdcnt   --alpha-key 1

