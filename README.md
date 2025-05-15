# wan2.1basecode
推理 python VideoX-Fun/examples/wan2.1_fun/predict_v2v_control.py  
在py文件里面设置ref——image，control video 路径，写prompt，为了批量测试我已做了修改，输入数据根目录即可 

训练 sh VideoX-Fun/scripts/wan2.1_fun/train_control.sh 在dataset里面设置好数据

[
  {
      "file_path": "/home/zhangjiaju/deeplearning/VideoX-Fun/sample/start_image/0001.png",
      "control_file_path" : "/home/zhangjiaju/deeplearning/VideoX-Fun/datasets/internal_datasets/train/control_imge/image.png",
      "text": "在这个阳光明媚的户外花园里，美女身穿一袭及膝的白色无袖连衣裙，裙摆在她轻盈的舞姿中轻柔地摆动，宛如一只翩翩起舞的蝴蝶。阳光透过树叶间洒下斑驳的光影，映衬出她柔和的脸庞和清澈的眼眸，显得格外优雅。仿佛每一个动作都在诉说着青春与活力，她在草地上旋转，裙摆随之飞扬，仿佛整个花园都因她的舞动而欢愉。周围五彩缤纷的花朵在微风中摇曳，玫瑰、菊花、百合，各自释放出阵阵香气，营造出一种轻松而愉快的氛围。",
      "type": "image"
  },
  {
    "file_path": "/home/zhangjiaju/deeplearning/VideoX-Fun/sample/start_image/0001.png",
    "control_file_path" : "/home/zhangjiaju/deeplearning/VideoX-Fun/datasets/internal_datasets/train/control_imge/image.png",
    "text": "在这个阳光明媚的户外花园里，美女身穿一袭及膝的白色无袖连衣裙，裙摆在她轻盈的舞姿中轻柔地摆动，宛如一只翩翩起舞的蝴蝶。阳光透过树叶间洒下斑驳的光影，映衬出她柔和的脸庞和清澈的眼眸，显得格外优雅。仿佛每一个动作都在诉说着青春与活力，她在草地上旋转，裙摆随之飞扬，仿佛整个花园都因她的舞动而欢愉。周围五彩缤纷的花朵在微风中摇曳，玫瑰、菊花、百合，各自释放出阵阵香气，营造出一种轻松而愉快的氛围。",
    "type": "image"
  }
]

写一个这样的格式的json，数据地址和上面写的一致

