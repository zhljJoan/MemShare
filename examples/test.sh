#torchrun --standalone --nproc-per-node 1 train_tgnn.py --dataname tgbl-wiki --model TGN > tgbl_wiki_train.out &
#wait
#torchrun --standalone --nproc-per-node 4 train_tgnn.py --dataname tgbl-wiki --model TGN > tgbl_wiki_train_4.out &
#wait
#torchrun --standalone --nproc-per-node 1 train_tgnn.py --dataname tgbl-review --model TGN > tgbl_review_train.out &
#wait
#torchrun --standalone --nproc-per-node 4 train_tgnn.py --dataname tgbl-review --model TGN > tgbl_review_train_4.out &
#wait
torchrun --standalone --nproc-per-node 1 train_tgnn.py --dataname tgbl-coin --model TGN_600 > tgbl_coin_train.out &
wait
torchrun --standalone --nproc-per-node 4 train_tgnn.py --dataname tgbl-coin --model TGN_600 > tgbl_coin_train_4.out &
wait
torchrun --standalone --nproc-per-node 1 train_tgnn.py --dataname tgbl-comment --model TGN_600 > tgbl_comment_train.out &
wait
torchrun --standalone --nproc-per-node 4 train_tgnn.py --dataname tgbl-comment --model TGN_600 > tgbl_comment_train_4.out &
wait
torchrun --standalone --nproc-per-node 1 train_tgnn.py --dataname tgbl-flight --model TGN_600 > tgbl_flight_train.out &
wait
torchrun --standalone --nproc-per-node 4 train_tgnn.py --dataname tgbl-flight --model TGN_600 > tgbl_flight_train_4.out &
wait
#torchrun --standalone --nproc-per-node 1 evaluate_tgbl_predict.py --dataname tgbl-wiki --model TGN --train_world_size 1 > tgbl_wiki_4.out &
#wait
#torchrun --standalone --nproc-per-node 1 evaluate_tgbl_predict.py --dataname tgbl-wiki --model TGN --train_world_size 4 > tgbl_wiki.out &
#wait
torchrun --standalone --nproc-per-node 1 evaluate_tgbl_predict.py --dataname tgbl-review --model TGN --train_world_size 1 > tgbl_review.out &
wait
torchrun --standalone --nproc-per-node 1 evaluate_tgbl_predict.py --dataname tgbl-review --model TGN --train_world_size 4 > tgbl_review_4.out &
wait
torchrun --standalone --nproc-per-node 1 evaluate_tgbl_predict.py --dataname tgbl-coin --model TGN_600 --train_world_size 1 > tgbl_coin.out &
wait
torchrun --standalone --nproc-per-node 1 evaluate_tgbl_predict.py --dataname tgbl-coin --model TGN_600 --train_world_size 4 > tgbl_coin_4.out &
wait
torchrun --standalone --nproc-per-node 1 evaluate_tgbl_predict.py --dataname tgbl-comment --model TGN_600 --train_world_size 1 > tgbl_comment.out &
wait
torchrun --standalone --nproc-per-node 1 evaluate_tgbl_predict.py --dataname tgbl-comment --model TGN_600 --train_world_size 4 > tgbl_comment_4.out &
wait
torchrun --standalone --nproc-per-node 1 evaluate_tgbl_predict.py --dataname tgbl-flight --model TGN_600 --train_world_size 1 > tgbl_flight.out &
wait
torchrun --standalone --nproc-per-node 1 evaluate_tgbl_predict.py --dataname tgbl-flight --model TGN_600 --train_world_size 4 > tgbl_flight_4.out &
wait