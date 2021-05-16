import random
from tensorflow import keras
import numpy as np
import cv2
import PIL
import Augmentor


def gaussian_noise(image, prob_of_applying_noise):
    noisy_image = np.zeros(image.shape, np.uint8)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if random.uniform(0,1) < prob_of_applying_noise:
                noisy_image[i][j] = image[i][j] + 255 * np.random.normal(0, 1)
            else:
                noisy_image[i][j] = image[i][j]

    return noisy_image

def gaussian_noise_both(image, target, prob_of_applying_noise):
    noisy_image = np.zeros(image.shape, np.uint8)
    noisy_target = np.zeros(target.shape, np.uint8)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if random.uniform(0, 1) < prob_of_applying_noise:
                gaussian = 255 * np.random.normal(0, 1)
                noisy_image[i][j] = image[i][j] + gaussian
                noisy_target[i][j] = target[i][j] + gaussian
            else:
                noisy_image[i][j] = image[i][j]
                noisy_target[i][j] = target[i][j]

    return noisy_image, noisy_target


def ripple(image, prob):
    noisy_image = np.zeros(image.shape, np.uint8)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            s = (((float(x)-((float(image.shape[1]))/2))**2)+((float(y)-((float(image.shape[0]))/2))**2))**(1/2)
            prob_of_applying_noise = prob * 2.71**(-(s/((image.shape[1])/3)))
            d = random.randint(7, 12)
            rem = (s/d)-(s//d)
            if random.uniform(0, 1) < prob_of_applying_noise:
                if rem < 0.5:
                    noisy_image[y][x] = 255 - ((255 - image[y][x]) * (1-(1.4*(rem-0.5))**2))
                if rem >= 0.5:
                    noisy_image[y][x] = (1-(1.4*(rem-0.5))**2) * image[y][x]
            else:
                noisy_image[y][x] = image[y][x]

    return noisy_image

def ripple_both(image, target, prob):
    noisy_image = np.zeros(image.shape, np.uint8)
    noisy_target = np.zeros(image.shape, np.uint8)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            s = (((float(x)-((float(image.shape[1]))/2))**2)+((float(y)-((float(image.shape[0]))/2))**2))**(1/2)
            prob_of_applying_noise = prob * 2.71**(-(s/((image.shape[1])/3)))
            d = random.randint(7, 12)
            rem = (s/d)-(s//d)
            if random.uniform(0, 1) < prob_of_applying_noise:
                if rem < 0.5:
                    noisy_image[y][x] = 255 - ((255 - image[y][x]) * (1-(1.4*(rem-0.5))**2))
                    noisy_target[y][x] = 255 - ((255 - target[y][x]) * (1 - (1.4 * (rem - 0.5)) ** 2))
                if rem >= 0.5:
                    noisy_image[y][x] = (1-(1.4*(rem-0.5))**2) * image[y][x]
                    noisy_target[y][x] = (1 - (1.4 * (rem - 0.5)) ** 2) * target[y][x]
            else:
                noisy_image[y][x] = image[y][x]
                noisy_target[y][x] = target[y][x]

    return noisy_image, noisy_target


class DataGen(keras.utils.Sequence):
    def __init__(self, index_list, image_paths, target_paths, o_w, o_h, t_w, t_h, rp_prob, in_rp, tar_rp, s_p, in_s, tar_s,  r_d, in_r, tar_r,
                 t_p, in_t, tar_t,  in_f, tar_f,  d_n, d_s, in_d, tar_d,  e_p, in_e, tar_e,  b_p, in_b, tar_b,  n_p, in_n, tar_n,
                 b_u, in_br, tar_br, c_u, in_c, tar_c, deep_sup=0,
                 batch_size=1):
        self.index_list = index_list
        self.image_paths = image_paths
        self.target_paths = target_paths
        self.o_w = o_w
        self.o_h = o_h
        self.t_w = t_w
        self.t_h = t_h

        self.rp_prob = rp_prob
        self.in_rp = in_rp
        self.tar_rp = tar_rp

        self.s_p = s_p
        self.in_s = in_s
        self.tar_s = tar_s

        self.r_d = r_d
        self.in_r = in_r
        self.tar_r = tar_r

        self.t_p = t_p
        self.in_t = in_t
        self.tar_t = tar_t

        self.in_f = in_f
        self.tar_f = tar_f


        self.d_n = d_n
        self.d_s = d_s
        self.in_d = in_d
        self.tar_d = tar_d

        self.e_p = e_p
        self.in_e = in_e
        self.tar_e = tar_e

        self.b_p = b_p
        self.in_b = in_b
        self.tar_b = tar_b

        self.n_p = n_p
        self.in_n = in_n
        self.tar_n = tar_n

        self.b_u = b_u
        self.in_br = in_br
        self.tar_br = tar_br

        self.c_u = c_u
        self.in_c = in_c
        self.tar_c = tar_c


        self.batch_size = batch_size
        self.deep_sup = deep_sup
        self.on_epoch_end()

    def __load__(self, load_image_index):
        ## Path
        image_path = self.image_paths[load_image_index]
        target_path = self.target_paths[load_image_index]

        ## Reading Image
        image = cv2.imread(image_path)
        image = cv2.resize(image, (self.o_w, self.o_h))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ## Reading Mask
        target = cv2.imread(target_path)
        target = cv2.resize(target, (self.o_w, self.o_h))
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        ## Image and mask augmentation with random perspective scalling, translation, and rotation
        height, width = self.o_h, self.o_w

        ## Set ranges
        frame = self.s_p
        theta = self.r_d
        hor = self.t_p
        ver = self.t_p

        ## Generate rectangles for dropout
        rectangles = {}
        for i in range(0, self.d_n):
            x = random.randint(0, width - self.d_s)
            y = random.randint(0, width - self.d_s)
            rectangles['point%s' % (i * 2)] = (x, y)
            rectangles['point%s' % (i * 2 + 1)] = (x + random.randint(0,self.d_s), y + random.randint(0,self.d_s))

        ## Generate random variables
        theta = random.uniform(-theta, theta)
        x1 = random.uniform(-frame, frame)
        y1 = random.uniform(-frame, frame)

        x2 = random.uniform(width - frame, width + frame)
        y2 = random.uniform(-frame, frame)

        x3 = random.uniform(-frame, frame)
        y3 = random.uniform(height - frame, height + frame)

        x4 = random.uniform(width - frame, width + frame)
        y4 = random.uniform(height - frame, height + frame)

        hor = random.uniform(-hor, hor)
        ver = random.uniform(-ver, ver)


        h_flip = random.randint(0, 1)

        v_flip = random.randint(0, 1)

        blur_rad = random.randint(0, self.b_p)

        ## Transform image
        # Ripple
        if self.in_rp == 1 and self.tar_rp == 0:
            image = ripple(image, self.rp_prob)

        # Rotation
        if self.in_r == 1:
            M = cv2.getRotationMatrix2D(((width - 1) / 2.0, (height - 1) / 2.0), theta, 1)
            image = cv2.warpAffine(image, M, (width, height))
        else:
            pass

        # Scaling
        if self.in_s == 1:
            pts1 = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
            pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
            M = cv2.getPerspectiveTransform(pts1, pts2)
            image = cv2.warpPerspective(image, M, (width, height))
        else:
            pass

        # Translation
        if self.in_t == 1:
            M = np.float32([[1, 0, hor], [0, 1, ver]])
            image = cv2.warpAffine(image, M, (width, height))

        # Flip
        if self.in_f ==1:
            if h_flip == 1:
                image = cv2.flip(image, 0)
            else:
                pass

            if v_flip == 1:
                image = cv2.flip(image, 1)
            else:
                pass
        else:
            pass

        color = (0, 0, 0)
        thickness = -1

        # Rectangular dropout
        if self.in_d == 1:
            for n in range(0, self.d_n):
                pt1 = rectangles['point%s' % (n * 2)]
                pt2 = rectangles['point%s' % (n * 2 + 1)]
                image = cv2.rectangle(image, pt1, pt2, color, thickness)


        # Elastic transform
        if self.in_e == 1 and self.tar_e == 0:
            images = [[image]]
            p = Augmentor.DataPipeline(images)
            p.random_distortion(probability=1, grid_width=int(self.o_w/7), grid_height=int(self.o_h/7), magnitude=self.e_p)
            a_images = p.sample(1)
            image = a_images[0][0]
        else:
            pass

        # Gaussian blur
        if self.in_b == 1 and blur_rad > 0:
            image = cv2.blur(image,(blur_rad, blur_rad))
        else:
            pass

        # Gaussian noise
        if self.in_n == 1 and self.tar_n == 0:
            image = gaussian_noise(image,random.uniform(0,self.n_p))
        else:
            pass

        # Contrast
        if self.in_c == 1 and self.tar_c == 0:
            factor = random.uniform(1, self.c_u)
            increase_decrease = random.randint(0,1)
            if increase_decrease == 0:
                enh = 1 / factor**3
            else:
                enh = factor
            image = keras.preprocessing.image.array_to_img(image)
            image = PIL.ImageEnhance.Contrast(image).enhance(enh)
            image = np.array(image)

        # Brightness
        if self.in_br == 1 and self.tar_br == 0:
            factor = random.uniform(1, self.b_u)
            increase_decrease = random.randint(0,1)
            if increase_decrease == 0:
                enh = 1 / factor**3
            else:
                enh = factor
            image = keras.preprocessing.image.array_to_img(image)
            image = PIL.ImageEnhance.Brightness(image).enhance(enh)
            image = np.array(image)




        ## Transform mask
        # Ripple
        if self.in_rp == 0 and self.tar_rp == 1:
            target = ripple(target, self.rp_prob)

        # Rotation
        if self.tar_r == 1:
            M = cv2.getRotationMatrix2D(((width - 1) / 2.0, (height - 1) / 2.0), theta, 1)
            target = cv2.warpAffine(target, M, (width, height))
        else:
            pass

        # Scaling
        if self.tar_s == 1:
            pts1 = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
            pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
            M = cv2.getPerspectiveTransform(pts1, pts2)
            target = cv2.warpPerspective(target, M, (width, height))
        else:
            pass

        # Translation
        if self.tar_t == 1:
            M = np.float32([[1, 0, hor], [0, 1, ver]])
            target = cv2.warpAffine(target, M, (width, height))

        # Flip
        if self.tar_f ==1:
            if h_flip == 1:
                target = cv2.flip(target, 0)
            else:
                pass

            if v_flip == 1:
                target = cv2.flip(target, 1)
            else:
                pass
        else:
            pass

        # Rectangular dropout
        if self.tar_d == 1:
            for n in range(0, self.d_n):
                pt1 = rectangles['point%s' % (n * 2)]
                pt2 = rectangles['point%s' % (n * 2 + 1)]
                target = cv2.rectangle(target, pt1, pt2, color, thickness)

        # Elastic transform
        if self.in_e == 0 and self.tar_e == 1:
            target = [[target]]
            p = Augmentor.DataPipeline(target)
            p.random_distortion(probability=1, grid_width=int(self.o_w / 7), grid_height=int(self.o_h / 7),
                                magnitude=self.e_p)
            a_target = p.sample(1)
            target = a_target[0][0]
        else:
            pass

        # Gaussian blur
        if self.tar_b == 1 and blur_rad > 0:
            target = cv2.blur(target, (blur_rad, blur_rad))
        else:
            pass

        # Gaussian noise
        if self.in_n == 0 and self.tar_n == 1:
            target = gaussian_noise(target, random.uniform(0,self.n_p))
        else:
            pass

        # Contrast
        if self.in_c == 0 and self.tar_c == 1:
            factor = random.uniform(1, self.c_u)
            increase_decrease = random.randint(0,1)
            if increase_decrease == 0:
                enh = 1 / factor**3
            else:
                enh = factor
            target = keras.preprocessing.image.array_to_img(target)
            target = PIL.ImageEnhance.Contrast(target).enhance(enh)
            target = np.array(target)

        # Brightness
        if self.in_br == 0 and self.tar_br == 1:
            factor = random.uniform(1, self.b_u)
            increase_decrease = random.randint(0,1)
            if increase_decrease == 0:
                enh = 1 / factor**3
            else:
                enh = factor
            target = keras.preprocessing.image.array_to_img(target)
            target = PIL.ImageEnhance.Brightness(target).enhance(enh)
            target = np.array(target)



        ### Apply to both
        # Ripple
        if self.in_rp == 1 and self.tar_rp == 1:
            image, target = ripple_both(image, target, self.rp_prob)

        # Elastic transform
        if self.in_e == 1 and self.tar_e == 1:
            images = [[image, target]]
            p = Augmentor.DataPipeline(images)
            p.random_distortion(probability=1, grid_width=int(self.o_w / 7), grid_height=int(self.o_h / 7),
                                magnitude=self.e_p)
            a_images = p.sample(1)
            image = a_images[0][0]
            target = a_images[0][1]
        else:
            pass

        # Gaussian noise
        if self.in_n == 1 and self.tar_n == 1:
            image, target = gaussian_noise_both(image, target, random.uniform(0,self.n_p))

        # Contrast
        if self.in_c == 1 and self.tar_c == 1:
            factor = random.uniform(1, self.c_u)
            increase_decrease = random.randint(0,1)
            if increase_decrease == 0:
                enh = 1 / factor**3
            else:
                enh = factor

            image = keras.preprocessing.image.array_to_img(image)
            target = keras.preprocessing.image.array_to_img(target)
            image = PIL.ImageEnhance.Contrast(image).enhance(enh)
            target = PIL.ImageEnhance.Contrast(target).enhance(enh)
            image = np.array(image)
            target = np.array(target)

        # Brightness
        if self.in_br == 1 and self.tar_br == 1:
            factor = random.uniform(1, self.b_u)
            increase_decrease = random.randint(0,1)
            if increase_decrease == 0:
                enh = 1/factor**3
            else:
                enh = factor
            image = keras.preprocessing.image.array_to_img(image)
            target = keras.preprocessing.image.array_to_img(target)
            image = PIL.ImageEnhance.Brightness(image).enhance(enh)
            target = PIL.ImageEnhance.Brightness(target).enhance(enh)
            image = np.array(image)
            target = np.array(target)

        #target = cv2.resize(target, (self.t_w, self.t_h))
        #target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        ## Normalizaing
        image = image / 255.0
        target = target / 255.0

        return image, target

    def __getitem__(self, index):
        if (index + 1) * self.batch_size > len(self.index_list):
            self.batch_size = len(self.ids) - index * self.batch_size

        Index_batch = self.index_list[index * self.batch_size: (index + 1) * self.batch_size]

        image = []
        target = []

        for load_image_index in Index_batch:
            _img, _target = self.__load__(load_image_index)
            image.append(_img)
            target.append(_target)


        if self.deep_sup == 1:
            image = np.array(image)
            target = np.array(target)
            target2 = np.array(target)
            targetconcat = np.concatenate((target, target2), axis=1)
            target = targetconcat

        if self.deep_sup == 0:
            image = np.array(image)
            target = np.array(target)

        return image, target

    def on_epoch_end(self):
        pass

    def __len__(self):
        return int(np.ceil(len(self.index_list) / float(self.batch_size)))
