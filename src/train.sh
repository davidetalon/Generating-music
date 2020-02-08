python src/train.py \
--wgan=1 \
--seed=30 \
--gen_lr=0.0001 \
--discr_lr=0.0001 \
--batch_size=16 \
--num_epochs=1 \
--attention=1 \
--save_interleaving=100 \
--extended_seq=0 \
--post_proc=1 \
--phase_shift=2 \
--data_dir='dataset/' \
# --model_folder='models' \
