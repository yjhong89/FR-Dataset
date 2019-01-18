import argparse
import struct
import logging
import numpy as np
import os
import cv2
from glob import glob
from PIL import Image
from face_recognition import FaceRecognition

class GEN_MEGAFACE(object):
    def __init__(self, args):
        self.args = args

        self.fr = FaceRecognition(model_file=self.args.ckpt, embedding_size=self.args.feature_dim)
        logging.info('Face recognition loaded')
        
        self.parent_path = './MEGAFACE'
        self.fname2center = dict()
        self.facescrub_noise = list()
        self.distractor_noise = list()

    def generate_filelist(self, path, noise_file):
        path = os.path.join(self.parent_path, path)
        logging.info('%s' % path)

        noise_list = list()
        with open(noise_file, 'r') as noise_f:
            while True:
                line = noise_f.readline()
                if not line:
                    break
                target = line.split('.')[0].strip()
                noise_list.append(target)
        logging.info('Check %s' % noise_list) 
            
        img_files = list()

        for (p,d,files) in os.walk(path):
            if len(files) == 0:
                continue
     
            for f in files:
                filename, file_ext = os.path.splitext(f)
                file_ext_lower = file_ext[1:].lower()

                if file_ext_lower in self.args.img_ext:
                    logging.info('%s added' % filename)
                    img_files.append(os.path.join(p, f))

        return img_files, noise_list       
    
    def generate_bin(self, img_path, bin_path, save_path, noise_list, megaface=True):
        logging.info('For %s' % img_path)
        path, filename = os.path.split(img_path)
        path, identity = os.path.split(path)
        path, parent_id = os.path.split(path)
        filename_head = os.path.splitext(filename)[0].strip()
        logging.debug('Filename: %s\nIdentity: %s\nPath: %s' % (filename_head, identity, path))

        if megaface:
            output_bin_parent_dir = os.path.join(save_path, bin_path, parent_id)
        else:
            output_bin_parent_dir = os.path.join(save_path, bin_path)
        output_bin_dir = os.path.join(output_bin_parent_dir, identity)
            
        if not os.path.exists(output_bin_dir):
            os.makedirs(output_bin_dir, exist_ok=True)
        logging.debug('Output directory: %s' % output_bin_dir)

        output_bin_path = os.path.join(output_bin_dir, filename_head + self.args.file_ending)

        img_ext = os.path.splitext(img_path)[-1]
        if img_ext == '.gif':
            gif = cv2.VideoCapture(img_path)
            _, img = gif.read()
        else:         
            img = cv2.imread(img_path)

        if img is None:
            raise Exception('%s not valid' % img_path)
        
        self.fr.forward([Image.fromarray(img[:,:,::-1])])
        feature = self.fr.numpy()
        feature = np.squeeze(feature, 1)

        if megaface:
            self.write_bin(output_bin_path, feature)
        else:
            self.facescrub_write_bin(output_bin_path, noise_list, filename_head, feature, identity)


    def megaface_write_bin(self, output_bin_path, noise_list, filename, feature, identity, parent_id):
        logging.debug('Filename: %s' % filename)
        noise_path = os.path.join(parent_id, identity, filename + '.jpg')
        if not noise_path in noise_list:
            feature_ = np.full((self.args.feature_dim+self.args.feature_ext), 0, dtype=np.float32)
            feature_[0:self.args.feature_dim] = feature
            self.write_bin(output_bin_path, feature)
        else:
            feature_ = np.full((self.args.feature_dim+self.args.feature_ext), 100, dtype=np.float32)
            feature_[0:self.args.feature_dim] = feature
            self.write_bin(output_bin_path, feature) 
            self.distractor_noise.append(output_bin_path)

    def write_bin(self, path, feature):
        features = list(feature)
        
        with open(path, 'wb') as f:
            f.write(struct.pack('4i', len(features),1,4,5))
            f.write(struct.pack('f'*len(features), *features)) 

        logging.info('Save done in %s' % path)

    def facescrub_write_bin(self, output_bin_path, noise_list, filename, feature, identity):        
        logging.debug('Filename: %s' % filename)
        if not filename in noise_list:
            feature_ = np.full((self.args.feature_dim+self.args.feature_ext), 0, dtype=np.float32)
            feature_[0:self.args.feature_dim] = feature
            self.write_bin(output_bin_path, feature)
            if not identity in self.fname2center:
                self.fname2center[identity] = np.zeros((self.args.feature_dim+self.args.feature_ext), dtype=np.float32)
            self.fname2center[identity] += feature_
        else:
            self.facescrub_noise.append((identity, filename, output_bin_path))
            logging.debug('%s added to facescrub_noise' % filename)

    def facescrub_noise_write_bin(self):
        logging.info('Facescrub noise length: %d' % len(self.facescrub_noise))
        for k in self.facescrub_noise:
            identity, filename, output_bin_path = k
            assert identity in self.fname2center
            center = self.fname2center[identity]
            g = np.zeros((self.args.feature_dim+self.args.feature_ext), dtype=np.float32)
            g2 = np.random.uniform(-0.001, 0.001, (self.args.feature_dim))
            g[0:self.args.feature_dim] = g2
            f = center + g
            _norm = np.linalg.norm(f)
            f /= _norm
            self.write_bin(output_bin_path, f)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--megaface_path', type=str)
    p.add_argument('--facescrub_path', type=str)
    p.add_argument('--megaface_noise', type=str)
    p.add_argument('--facescrub_noise', type=str)
    p.add_argument('--megascrub_bin_path', type=str)
    p.add_argument('--facescrub_bin_path', type=str)
    p.add_argument('--ckpt', type=str)
    p.add_argument('--img_ext', nargs='+')
    p.add_argument('--noise', action='store_true')
    p.add_argument('--file_ending', type=str)
    p.add_argument('--feature_ext', type=int)
    p.add_argument('--feature_dim', type=int)

    p.set_defaults(megaface_path='aligned_distractors',
                    facescrub_path='aligned_facescrub',
                    megaface_noise='./distractor_noise_list.txt',
                    facescrub_noise='./facescrub_noise_list.txt',
                    megaface_bin_path='megaface_bin',
                    facescrub_bin_path='facescrub_bin',
                    ckpt='_weights/net_epoch_0455_acc0.9614_thd0.165000.pth',
                    img_ext=['jpg', 'png', 'jpeg', 'gif'],
                    file_ending='_baseline.bin',
                    feature_ext=1,
                    feature_dim=512)

    args = p.parse_args()

    logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s][%(filename)s:%(lineno)d] %(message)s')

    if not os.path.exists(args.megaface_bin_path):
        os.makedirs(args.megaface_bin_path, exist_ok=True)
    if not os.path.exists(args.facescrub_bin_path):
        os.makedirs(args.facescrub_bin_path, exist_ok=True)

    mg = GEN_MEGAFACE(args)
    
    try:    
        megaface_files, megaface_noise = mg.generate_filelist(args.megaface_path, args.megaface_noise)
        logging.info('#files: %d' % len(megaface_files))
        facescrub_files, facescrub_noise = mg.generate_filelist(args.facescrub_path, args.facescrub_noise) 
        logging.info('#files: %d' % len(facescrub_files))
    except Exception as e:
        logging.error(e)
        raise

    success_counter = 0
    error_counter = 0
    save_path = args.file_ending.split('.')[0]   
 
    for f in megaface_files:
        try:
            mg.generate_bin(f, args.megaface_bin_path, save_path, megaface_noise, megaface=True)        
            success_counter += 1

        except Exception as e:
            logging.warn(e)
            error_counter += 1
            raise

        finally:
            logging.info('#success: %d, #fail: %d \n' % (success_counter, error_counter))

    logging.info('#Distractor noise: %d' % len(mg.distractor_noise))

    success_counter = 0
    error_counter = 0

    for f in facescrub_files:
        try:
            mg.generate_bin(f, args.facescrub_bin_path, save_path, facescrub_noise, megaface=False)        
            success_counter += 1

        except Exception as e:
            logging.warn(e)
            error_counter += 1
            raise

        finally:
            logging.info('#success: %d, #fail: %d\n' % (success_counter, error_counter))

    try:
        logging.info('Write noise files')
        mg.facescrub_noise_write_bin()
        logging.info('Finish facescrub bin files')
    except Exception as e:
        logging.error(e)
        raise

    
