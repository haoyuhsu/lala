# train cliport on stack-towers
python cliport/eval.py model_task=stack-towers \
                       eval_task=stack-towers \
                       agent=cliport \
                       mode=val \
                       n_demos=100 \
                       train_demos=500 \
                       exp_folder=exps \
                       checkpoint_type=last \
                       update_results=True \
                       disp=False \
                       record.save_video=False

# train image_goal_transporter on stack-towers
python cliport/eval.py model_task=stack-towers \
                       eval_task=stack-towers \
                       agent=image_goal_transporter \
                       mode=val \
                       n_demos=100 \
                       train_demos=500 \
                       exp_folder=exps \
                       checkpoint_type=last \
                       update_results=True \
                       disp=False \
                       record.save_video=False

# train image_goal_transporter on stack-boxes
python cliport/eval.py model_task=stack-boxes \
                       eval_task=stack-boxes \
                       agent=image_goal_transporter \
                       mode=val \
                       n_demos=100 \
                       train_demos=500 \
                       exp_folder=exps \
                       checkpoint_type=last \
                       update_results=True \
                       disp=False \
                       record.save_video=False

# train two_stream_image_goal_transporter on stack-towers
python cliport/eval.py model_task=stack-towers \
                       eval_task=stack-towers \
                       agent=two_stream_image_goal_transporter \
                       mode=val \
                       n_demos=100 \
                       train_demos=500 \
                       exp_folder=exps \
                       checkpoint_type=last \
                       update_results=True \
                       disp=False \
                       record.save_video=False

# train two_stream_image_goal_transporter on stack-boxes
python cliport/eval.py model_task=stack-boxes \
                       eval_task=stack-boxes \
                       agent=two_stream_image_goal_transporter \
                       mode=val \
                       n_demos=100 \
                       train_demos=500 \
                       exp_folder=exps \
                       checkpoint_type=last \
                       update_results=True \
                       disp=False \
                       record.save_video=False

# train two_stream_image_goal_transporter_lat on stack-towers
python cliport/eval.py model_task=stack-towers \
                       eval_task=stack-towers \
                       agent=two_stream_image_goal_transporter_lat \
                       mode=val \
                       n_demos=100 \
                       train_demos=500 \
                       exp_folder=exps \
                       checkpoint_type=last \
                       update_results=True \
                       disp=False \
                       record.save_video=False

# train two_stream_image_goal_transporter_lat on stack-boxes
python cliport/eval.py model_task=stack-boxes \
                       eval_task=stack-boxes \
                       agent=two_stream_image_goal_transporter_lat \
                       mode=val \
                       n_demos=100 \
                       train_demos=500 \
                       exp_folder=exps \
                       checkpoint_type=last \
                       update_results=True \
                       disp=False \
                       record.save_video=False