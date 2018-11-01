import os
import shutil
import logging
import argparse
import urllib
import requests
import concurrent.futures
import csv
from PIL import Image
import io
import multiprocessing as mp
import pickle

def create_logger(logger_name):
    logger = logging.getLogger(logger_name)
    
    # Set log level
    logger.setLevel(logging.DEBUG)

    # Log formatter
    formatter = logging.Formatter('[%(levelname)s] %(message)s')

    # Logfile handler
    logfile_name = logger_name+'.log'
    if os.path.exists(logfile_name):
        os.remove(logfile_name)

    file_handler = logging.FileHandler(logfile_name)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Print log to console
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logger.addHandler(console)

    return logger


def down_save(row, logname, crop, counter):
    logger = logging.getLogger(logname)

    # Make download directory
    dir_path = row['index']
    if not os.path.exists(dir_path):
        logger.info('Make directory of {name}\n'.format(name=row['index']))
        os.makedirs(dir_path)
    # Download image
    response = download_image(row['url'], logger, counter)
    if response:
        # response.content: bytes type
        save_image(response.content, crop, dir_path, row['image'], row['rect'], row['height width'])


def save_image(content, crop, dir_path, image_name, bb, image_size):
    assert os.path.exists(dir_path)

    save_path = os.path.join(dir_path, image_name)            

    # Get size
    h, w = list(map(int, image_size.split(' ')))
    logger.info('Image size >> height: %d, width: %d' % (h, w))    
    
    # Get bounding box 
    bbox = list(map(int, bb.split(' ')))
    logger.info('Bounding box >> x0: {x0}, y0: {y0}, x1: {x1}, y1: {y1}'.format(x0=bbox[0], y0=bbox[1], x1=bbox[2], y1=bbox[3])) 

    if crop:
        try:
            I = Image.open(io.BytesIO(content))
            I.resize((w,h), Image.ANTIALIAS).crop(bbox).save(save_path)
            logger.info('Image saved as %s\n' % save_path)

        except IOError as e:
            logger.error('{error}: {image}\n'.format(error=e, image=save_path))
            pass
    else:
        logger.warning('Not crop\n')
         
 
def download_image(url, logger, counter):
    '''
        Download image from url
        Returns response object if successful else None
    '''
    try:
        r = requests.get(url, stream=True) 
        # 200: OK
        status = r.status_code

        if status != requests.codes.OK:
            logger.warning('Line {line}: Can not access to {url}'.format(line, counter, url=url))
            r.raise_for_status()
            return None

        total_size = int(r.headers.get('content-length'))
        logger.info('Line {line}: Get image size of {size} from {url} with status {status}'.format(url=url, size=total_size, status=status, line=counter))

        return r

    except IOError as e:
        logger.error('{error}: {url}\n'.format(error=e, url=url))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script for download IMDb dataset for face recognition')
    parser.add_argument('--csv_file', type=str, default='IMDb-Face.csv')
    parser.add_argument('--logger', type=str, default='imdb')
    parser.add_argument('-c', '--crop', action="store_true")
    parser.add_argument('-d', '--delete', action="store_true")
    args = parser.parse_args()
    
    logger = create_logger(args.logger) 
    if args.delete:
        logger.warning('Delete existing files')
        files = os.listdir()
        for i in range(len(files)):
            if os.path.isdir(files[i]):
                shutil.rmtree(files[i])
                logger.warning('Delete %s' % files[i])
            elif os.path.splitext(files[i])[-1] == '.log':
                os.remove(files[i])
                logger.warning('Delete logfile %s' % files[i])
            else:
                continue

    num_threads = mp.cpu_count()

    # Check if csv file exists
    if not os.path.exists(os.path.expanduser(args.csv_file)):
        logger.error('%s does not exist' % args.csv_file)     
        raise Exception

    # Multiprocess must be called from __main__
    with open(args.csv_file) as imdb, concurrent.futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
        csv_reader = csv.DictReader(imdb)

        headers = csv_reader.fieldnames
        # Name, index, image, rect, size, url
        name = headers[0]
        index = headers[1]
        image_name = headers[2]
        rect = headers[3]
        size = headers[4]
        url = headers[5]
        logger.info('Headers: {name}, {index}, {image_index}, {rect}, {size}, {url}\n'.format(name=name, index=index, image_index=image_name, rect=rect, size=size, url=url))

        # logger is a lock object incurs pickle thread lock error
        #pickle.dumps(logger)
        
        try:
            futures = [executor.submit(down_save, row, args.logger, args.crop, counter) for counter, row in enumerate(csv_reader)]
            for future in concurrent.futures.as_completed(futures):
                #print(i.result())
                pass
        except Exception as e:
            logger.warning(e)
