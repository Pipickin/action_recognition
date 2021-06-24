# Features extraction

First you need to extract features via human estimation:

$ cd AlphaPose
$ python video_demo.py --video=${video_path} --outdir=${output_dir} --save_video --vis_fast

Example:
$ python video_demo.py --video=videos/yoga_1.mp4 --outdir=crafted_videos/ --save_video --vis_fast

$ python video_demo.py --video=/media/shchetkov/HDD/media/videos/task3/not_jumping_cut/swim.mp4 --outdir=/media/shchetkov/HDD/media/videos/task3/not_jumping_points/ --outest=/media/shchetkov/HDD/media/videos/task3/not_jumping_points_only/ --save_video --vis_fast


You can extract features from all videos in the folder:
$ bash videos_estimate.sh


Create folder with images and labels
$ python video2frames.py --video_path=/media/shchetkov/HDD/media/videos/task3/not_jumping_points_only/swim.avi --label=0 --save_folder=./data/test/

For all videos folder
$ python video2frames.py --video_path=/media/shchetkov/HDD/media/videos/task3/not_jumping_points_only/ --save_folder=./data/test/ --all_folder=True --label=0


Predict:
1) You need to run 
