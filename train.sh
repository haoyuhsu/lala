# train cliport on stack-towers
python cliport/train.py \
    train.task=stack-towers \
    train.agent=cliport \
    train.n_demos=500 \
    train.n_steps=201000 \
    train.exp_folder=exps \
    dataset.cache=False \
    train.n_val=20 \
    dataset.feature_maps=False 

# train image_goal_transporter on stack-towers
python cliport/train.py \
    train.task=stack-towers \
    train.agent=image_goal_transporter \
    train.n_demos=500 \
    train.n_steps=201000 \
    train.exp_folder=exps \
    dataset.cache=False \
    train.n_val=20 \
    dataset.feature_maps=True 

# train image_goal_transporter on stack-boxes
python cliport/train.py \
    train.task=stack-boxes \
    train.agent=image_goal_transporter \
    train.n_demos=500 \
    train.n_steps=201000 \
    train.exp_folder=exps \
    dataset.cache=False \
    train.n_val=20 \
    dataset.feature_maps=True 

# train two_stream_image_goal_transporter on stack-towers
python cliport/train.py \
    train.task=stack-towers \
    train.agent=two_stream_image_goal_transporter \
    train.n_demos=500 \
    train.n_steps=201000 \
    train.exp_folder=exps \
    dataset.cache=False \
    train.n_val=20 \
    dataset.feature_maps=True 

# train two_stream_image_goal_transporter on stack-boxes
python cliport/train.py \
    train.task=stack-boxes \
    train.agent=two_stream_image_goal_transporter \
    train.n_demos=500 \
    train.n_steps=201000 \
    train.exp_folder=exps \
    dataset.cache=False \
    train.n_val=20 \
    dataset.feature_maps=True 

# train two_stream_image_goal_transporter_lat on stack-towers
python cliport/train.py \
    train.task=stack-towers \
    train.agent=two_stream_image_goal_transporter_lat \
    train.n_demos=500 \
    train.n_steps=201000 \
    train.exp_folder=exps \
    dataset.cache=False \
    train.n_val=20 \
    dataset.feature_maps=True 

# train two_stream_image_goal_transporter_lat on stack-boxes
python cliport/train.py \
    train.task=stack-boxes \
    train.agent=two_stream_image_goal_transporter_lat \
    train.n_demos=500 \
    train.n_steps=201000 \
    train.exp_folder=exps \
    dataset.cache=False \
    train.n_val=20 \
    dataset.feature_maps=True 