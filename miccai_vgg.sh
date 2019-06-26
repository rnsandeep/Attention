
max_epoch=60
classes=8
dataset='/home/ubuntu/miccai/trainval_split'

exp='miccai_without_attention_focal'
meanfile='mean_std_darkcircle.npy'

####### Training Network #########################

#python train_vgg.py $dataset $exp $meanfile 299 $classes $max_epoch

########## TEsting NEtwork #######################
max_size=224
min_size=224
for epoch in `seq 3 $max_epoch`
do
   for size in `seq $min_size 20 $max_size`	
   do	   
       f="_checkpoint.pth.tar"
       echo "epoch: ", $epoch, "resize_size: ", $size
       python test_vgg.py $dataset $exp/$epoch$f $meanfile 224 $size $classes out_$exp
   done    
done

