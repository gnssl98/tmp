- DATADOWNLOAD (OPTIONS)
공개된 학습 데이터 MJ ST데이터를 다운 받습니다 (40G)  
압축을 푼뒤 data_set/training으로 이동 합니다.
1. https://www.dropbox.com/sh/i39abvnefllx2si/AAAbAYRvxzRp3cIE5HzqUw3ra?dl=0&preview=data_lmdb_release.zip
2. https://www.dropbox.com/sh/i39abvnefllx2si/AAAbAYRvxzRp3cIE5HzqUw3ra?dl=0&preview=ST_spe.zip

```bash
unzip -q data_lmdb_release.zip
cp -R data_lmdb_release/training/* ./data_set/training/
mv ST_spe.zip ./data_set/training/

DOCKER START
sudo docker build -t text_rec_v1 -f docker/Dockerfile .
sudo docker run -itd --name pytorch -v $(pwd):/workspace -p 8888:8888 --gpus all text_rec_v1 /bin/bash
sudo docker exec -i -t ${CONTAINER ID}

$(bash) cd workspace
$(bash) conda env create --file text_recognition_package.yaml

데이터 정제
data
├── gt.txt
└── /path/to/images
    ├── images_1.png
    ├── images_2.png
    ├── images_3.png
    └── ...
gt.txt 파일의 구성은 {imagepath}\t{label}\n 이와 같이 구성 합니다.

/path/to/images/images_1.png 한글
/path/to/images/images_2.png 사랑

python ./utils/lmdb_convert.py --inputPath /path/to/images --gtFile /path/to/gt.txt --outputPath result/{sub_dir}
```

모델학습 
```` bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --train_data ./result/ --valid_data ./validation_result --select_data global --batch_ratio 1.0
```

 with open(f'./saved_models/{opt.exp_name}/opt.txt', 'a') as opt_file:
        opt_log = '------------ Options -------------\n'
        args = vars(opt)
        for k, v in args.items():
            opt_log += f'{str(k)}: {str(v)}\n'
        opt_log += '---------------------------------------\n'
        print(opt_log)
        opt_file.write(opt_log)



