python src/train.py \
--wgan=1 \
--seed=30 \
--gen_lr=0.0001 \
--discr_lr=0.0001 \
--batch_size=64 \
--num_epochs=3000 \
--attention=1 \
--notes='wgan/1e-4/3000/64/att/no_ext/postproc' \
--extended_seq=0 \
--post_proc=1 \
# --model_path='29-01-2020_15-22-18' \