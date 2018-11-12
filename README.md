# Face Recognition dataset

## IMDb-Face dataset

* [IMDb](http://openaccess.thecvf.com/content_ECCV_2018/papers/Liren_Chen_The_Devil_of_ECCV_2018_paper.pdf) 
  + Available at https://github.com/fwang91/IMDb-Face (IMDb-Face.csv)
  
* For fast processing, multi processing is supported with concurrent.futures module.
  
## Running instruction
* Download 'IMDb-Face.csv' file from https://drive.google.com/open?id=134kOnRcJgHZ2eREu8QRi99qj996Ap_ML
* Run     
    ```
    imdb_crawl.py
    -c: whether you crop the image with bounding box
    -d: delete existing data directory be
    ```
* If you save non-cropped image, corresponding bounding box will also be recorded in bb.txt file for each direcory.
* Make sure 'IMDb-Face.csv' and 'imdb_crawl.py' are located in same directory.

## Megaface dataset
