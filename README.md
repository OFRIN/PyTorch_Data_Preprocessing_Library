
"OpenCV Encode -> Decode" vs "Pillow Encode -> Decode (ByteIO)" 비교

"convert_OpenCV_to_PIL"의 단점 (속도 저하 원인)

```
# 129sec - 7,500 images
python make_dataset.py --save_dir ./Example_100/ --the_number_of_image_per_file 100

# 42sec - 7,500 images
python make_dataset_using_ray.py --save_dir ./Example_100/ --the_number_of_image_per_file 100 --the_size_of_accumulation 1000

# 42sec - 7,500 images
python make_dataset_using_ray.py --save_dir ./Example_100/ --the_number_of_image_per_file 100 --the_size_of_accumulation 10000
```

```
# 42sec - 7,500 images
python make_dataset_using_ray.py --save_dir ./Example_100/ --the_number_of_image_per_file 100 --the_size_of_accumulation 1000

# 42sec - 7,500 images
python make_dataset_using_ray.py --save_dir ./Example_250/ --the_number_of_image_per_file 250 --the_size_of_accumulation 1000

# 42sec - 7,500 images
python make_dataset_using_ray.py --save_dir ./Example_500/ --the_number_of_image_per_file 500 --the_size_of_accumulation 1000

# 42sec - 7,500 images
python make_dataset_using_ray.py --save_dir ./Example_750/ --the_number_of_image_per_file 750 --the_size_of_accumulation 1000

# 42sec - 7,500 images
python make_dataset_using_ray.py --save_dir ./Example_1000/ --the_number_of_image_per_file 1000 --the_size_of_accumulation 1000
```

```
python ex_decoder_using_ray.py
```

```
python make_dataset_using_ray.py --save_dir C:/Classification_DB_PIL/ --the_number_of_image_per_file 250 --the_size_of_accumulation 1000
```