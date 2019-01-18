# Face Recognition dataset

## IMDb-Face dataset

* [IMDb](http://openaccess.thecvf.com/content_ECCV_2018/papers/Liren_Chen_The_Devil_of_ECCV_2018_paper.pdf) 
  + Available at https://github.com/fwang91/IMDb-Face (IMDb-Face.csv)
  
* For fast processing, multi processing is supported with concurrent.futures module.
  
### Running instruction
* Download 'IMDb-Face.csv' file from https://drive.google.com/open?id=134kOnRcJgHZ2eREu8QRi99qj996Ap_ML
* `python imdb_crawl.py`
  * Arguments
   ```   
   -c: whether you crop the image with bounding box
   -d: delete existing data directory be
   ```
* If you save non-cropped image, corresponding bounding box will also be recorded in bb.txt file for each direcory.
* Make sure 'IMDb-Face.csv' and 'imdb_crawl.py' are located in same directory.

## Megaface dataset
To run megaface test including identification(1m distractors), verification(@1e-6), 
1. Download distractors and probe dataset
2. Preprocess dataset
3. Generate bin files with a trained face recognition model
4. Run megaface devkit

### Download dataset
* Distractors<br/>
  `wget -c --user 'id' --password 'pwd' http://megaface.cs.washington.edu/dataset/download/content/MegaFace_dataset.tar.gz`
* Facescrub<br/>
  - `wget -c --user 'id' --password 'pwd' http://megaface.cs.washington.edu/dataset/download/content/downloaded.tgz`
  - `python facescrub.py`<br/>
    But, many urls can not be accessed.
* Both datasets can be accessed at **http://megaface.cs.washington.edu/participate/challenge.html**
* Dataset structure
   <pre>
   MEGAFACE -- distractors -- parent id -- ids -- images
            |                                  |- json file for each image 
            |
            |- facescrub -- ids -- images, bb.txt
                                |- bb.txt
   </pre>

### Preprocess
Preprocess with your face detection/alignment model.

### Generate bin files
* gen_megaface.py (Need your face recognition model)  
  - Make bin files of megaface distractors/facescrub images from trained face recognition model.
  - Arguments
    <pre>
    - megaface_path: path of pre-processed distractor images
    - facescrub_path: path of pre-processed facescrub images
    - megaface_noise: noise list of distractors
    - facescrub_noise: noise list of facescrub
    - megaface_bin_path: distractor bins save directory
    - facescrub_bin_path: facescrub bins save directory
    - ckpt: trained face recognition model
    - file_ending: file ending name, ex) _baseline.bin: aaa.jpg -> aaa_baseline.bin    
    </pre>
  - Resulting bin files of gen_megaface.py (ex: file_ending: \_baseline.bin_)
 
    <pre>
    _baseline -- facescrub_bin -- ids -- bin files (***_baseline.bin)
                             | 
                             |- megaface_bin -- parend id -- ids -- bin files (***_baseline.bin)
    </pre>
 
 
### Run megaface devkit
* On terminal, <br/>
``` python run_experiment.py --file_ending _baseline.bin --out_root baseline_results -d```
  - Need at least 32G memory
  - Need devkit from http://megaface.cs.washington.edu/participate/challenge.html
    - bin/FuseResults, Identification
    - models/jb_identity.bin, jb_LBP.bin
    - templatelists
    
* run_experiment.py
  - Executes identification and verification binary files (**bin/Identification**, **bin/FuseResults**).
  - Arguments
    <pre>
    - distractor_feature_path: distractor bin files path (<b>megaface_bin_path</b> of gen_megaface.py)
    - probe_feature_path: facescrub bin files path (<b>facescrub_bin_path</b> of gen_megaface.py)
    - file_ending: file ending format (<b>file_ending</b> of gen_megaface.py)
    - sizes: number of distractors, set as [1000000] 
    </pre>
* Caution
  - binary files (**bin/Identification**, **bin/FuseResults**) can be only executed on opencv2.4. 

## Reference
* Megaface devkit
  - Can download at http://megaface.cs.washington.edu/participate/challenge.html
* distractor_noise_list, facescrub_noise_list
  - Megaface dataset has some noisy dataset (some images in facescrub are included distractors) 
    - [arcface](https://arxiv.org/pdf/1801.07698.pdf)
    - [arcface github](https://github.com/deepinsight/insightface/tree/master/src/megaface)

