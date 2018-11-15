import os
import urllib
import requests
import argparse
from PIL import Image
import io
import logging
import multiprocessing as mp
import concurrent.futures 
import shutil

DATA_ROOT = './facescrub'

def downloads(line, timeout):
    # Name/image_id/face_id/url/bbox/sha256  
    # Line parse  
    contents = line.split('\t')
    act_name = contents[0].strip()
    image_id = contents[1].strip()
    url = contents[3].strip()
    bbox = contents[4].strip()
    
    download_dir = os.path.join(DATA_ROOT, act_name)
    
    if not os.path.exists(act_name):
        os.makedirs(act_name)
        logging.info('%s directory created' % act_name)

    try:
        r = requests.get(url, stream=True, timeout=timeout)
        status = r.status_code
        content_size = int(r.headers.get('content-length'))
        logging.info('Status {} for {}, size of {}'.format(status, save_path, content_size))
        
        if content_size == 0:
            return None
    
        I = Image.open(io.BytesIO(r.content))
        I.save(save_path)
        logging.info('%s saved\n' % save_path)
        # Write bounding box
        with open(os.path.join(download_dir, 'bb.txt'), 'a') as bbf:
            bbf.write(('_').join((act_name, face_id)) + ',' + bbox + '\n')
            bbf.close()

    except Exception as e:
        logging.error(', '.join((url, str(e))) + '\n') 
                

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script for download face scrub dataset')
    parser.add_argument('--txt_files', nargs='+')
    parser.add_argument('--timeout', type=int, default=10)
    parser.add_argument('-d', '--delete', action='store_true')

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    if args.delete:
        logging.warning('Delete existing images')
        files = os.listdir()
        for i in range(len(files)):
            if os.path.isdir(files[i]):
                try:
                    shutil.rmtree(files[i])
                    logging.warning('Delete %s' % files[i])
                except OSError as e:
                    logging.error('%s-%s' % (e.filename, e.strerror))
                else:
                    continue
    else:
        for (p, d, files) in os.walk(DATA_ROOT):
            for f in files:
                ext = os.path.splitext(f)[-1]
                if ext == '.txt':
                    os.remove(os.path.join(p, f))
                    logging.warning('Delete %s' % os.path.join(p, f))
    
    if not os.path.exists(DATA_ROOT):
        os.makedirs(DATA_ROOT)

    logging.info(args.txt_files)

    num_threads = mp.cpu_count()
    logging.info('%d cpu' % num_threads)

    for i in range(len(args.txt_files)):
        logging.info('Open %s' % args.txt_files[i])
        with open(args.txt_files[i]) as f, concurrent.futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
            lines = f.readlines()
            
            try:
                futures = [executor.submit(downloads, line, args.timeout) for line in lines[1:]]
                for future in concurrent.futures.as_completed(futures):
                    try:
                        data = future.result()
                    except Exception as exc:
                        logging.warning(exc)
#                    else:
#                        logging.info('%si bytes done' % data) 

            except Exception as e:
                logging.warning('%(lineno)d:', e)


            f.close()
            executor.shutdown(wait=True)
