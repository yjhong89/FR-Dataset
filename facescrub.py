import os
import urllib
import requests
import argparse
from PIL import Image
import io
import logging
import multiprocessing as mp
import subprocess as sp
import concurrent.futures 
import shutil

DATA_ROOT = './facescrub'

def downloads(line, timeout, downloaded):
    # Name/image_id/face_id/url/bbox/sha256  
    # Line parse  
    contents = line.split('\t')
    act_name = contents[0].strip().replace(' ', '_')
    face_id = contents[2].strip()
    url = contents[3].strip()
    bbox = contents[4].strip()

    download_dir = os.path.join(DATA_ROOT, act_name)

    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
        logging.info('%s directory created' % act_name)

    try:
        if not downloaded:
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
                
def clean_dir_file_name():
    directories = os.listdir(DATA_ROOT)

    for d in directories:
        d_changed = d.split('_')[0].strip().replace(' ', '_')
        if d != d_changed:
            sp.call(['mv', d, d_changed])
            logging.info('From {} to {}'.format(d, d_changed))
        files = os.listdir(os.path.join(DATA_ROOT, d_changed))
        for f in files:
            f_changed = d.split('_')[0].strip().replace(' ', '_')
            if f != f_changed:
                sp,call(['mv', os.path.join(d_changed, f), os.path.join(d_changed, f_changed)])
                logging.info('From {} to {}'.format(os.path.join(d_changed, f), os.path.join(d_changed, f_changed)))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script for download face scrub dataset')
    # ['facescrub_actors.txt', 'facescrub_actresses.txt']
    parser.add_argument('--txt_files', nargs='+')
    parser.add_argument('--timeout', type=int)
    parser.add_argument('-d', '--delete', action='store_true')
    parser.add_argument('--downloaded', action='store_true', help='Downloaded full tgz file or not')
    parser.add_argument('--datapath', type=str)
    parser.set_defaults(txt_files=['facescrub_actors.txt', 'facescrub_actresses.txt'], timeout=10, datapath='facescrub')

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    if args.delete:
        logging.warning('Delete existing images directory')
        try:
            shutil.rmtree(DATA_ROOT)
            logging.warning('Delete %s' % DATA_ROOT)
        except OSError as e:
            logging.error('%s-%s' % (e.filename, e.strerror))
   
    else:
        for (p, d, files) in os.walk(DATA_ROOT):
            for f in files:
                ext = os.path.splitext(f)[-1]
                if ext == '.txt':
                    os.remove(os.path.join(p, f))
                    logging.warning('Delete %s' % os.path.join(p, f)) 

        
    if not os.path.exists(DATA_ROOT):
        os.makedirs(DATA_ROOT)

    if args.downloaded:    
        clean_dir_file_name()
        num_threads = 1

    else:
        num_threads = mp.cpu_count()

    logging.info('Facescrub text files: ', args.txt_files)

    num_threads = 1
    logging.info('%d cpus' % num_threads)

    for i in range(len(args.txt_files)):
        logging.info('Open %s' % args.txt_files[i])
        with open(args.txt_files[i]) as f, concurrent.futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
            lines = f.readlines()
            
            try:
                futures = [executor.submit(downloads, line, args.timeout, args.downloaded) for line in lines[1:]]
                for future in concurrent.futures.as_completed(futures):
                    try:
                        data = future.result()
                    except Exception as exc:
                        logging.warning(exc)
#                    else:
#                        logging.info('%s bytes done' % data) 

            except Exception as e:
                logging.warning('%(lineno)d:', e)


            f.close()
            executor.shutdown(wait=True)
