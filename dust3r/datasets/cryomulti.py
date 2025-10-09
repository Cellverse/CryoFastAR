import torch
import os
import numpy as np
from glob import glob
from torch.utils.data import Dataset
from dust3r.datasets.base.easy_dataset import EasyDataset
from dust3r.utils.fft import ht2_center, symmetrize_ht
import itertools
from dust3r.datasets.ctf import compute_chi_from_drgn_ctfs
import pandas as pd
# rom torchvision.transforms import ToTensor

def valid_rotation(Rs, min_angle, max_angle):
    indices = [i for i in range(Rs.shape[0])]
    combinations = list(itertools.combinations(indices, 2))
    for combo in combinations:
        R1 = Rs[combo[0]]
        R2 = Rs[combo[1]]
        angle_deg = rotation_matrix_angle_difference(R1, R2)
        if min_angle and np.any(angle_deg < min_angle):
            return False
        if max_angle and np.any(angle_deg > max_angle):
            return False
    return True

def rotation_matrix_angle_difference(R1, R2):
    # Compute the relative rotation matrix
    R_rel = R1 @ R2.T
    # Compute the trace of the relative rotation matrix
    trace_R_rel = np.trace(R_rel)

    cosine = (trace_R_rel - 1) / 2
    cosine = cosine.clip(-1, 1)
    # Compute the angle difference
    angle =  np.arccos(cosine)
    if angle > 90:
        angle = 180 - angle

    # Convert the angle from radians to degrees (optional)
    angle_degrees = np.degrees(angle)
    return angle_degrees

class CryoMultiViewDataset(EasyDataset):
    def __init__(self, 
                 root_path, 
                 transform=None, 
                 seed=None, 
                 num_views=2, 
                 mini_batch_size=1, 
                 max_angle=None, 
                 min_angle=None, 
                 norm=None, 
                 mode='ht', 
                 per_scene=False, 
                 invert_data=False, 
                 apix_path=None, 
                 apix=None, 
                 image_scalar = 1.0, 
                 first_N = None,
                 skip_first_N = 0,
                 load_draco = True,
            ):

        self.root_path = root_path
        self.image_dirs = glob(os.path.join(root_path, '*'))
        # verify if the image_dirs is not empty and it is a real dir
        # self.image_dirs = [d for d in self.image_dirs if os.path.isdir(d)]
        assert len(self.image_dirs) > 0, f"No directories found in {root_path}"
        self.seed = seed
        self.num_views = num_views
        self.mini_batch_size = mini_batch_size
        self.cached_images = {}
        self.cached_poses = {}
        self.max_angle = max_angle
        self.min_angle = min_angle
        self.mode = mode
        self.invert_data = invert_data
        # FIXME: load an image for the image size
        self.im_size = (128, 128)
        if mode == 'ht':
            self.norm = norm or self.estimate_normalization()
        elif mode == 'unfft':
            self.ktraj = self.create_ktraj(spokelength=128, nspokes=128)
            self.grid_size = (128, 128) # (spokelength, spokelength)

            import torchkbnufft as tkbn
            self.nufft_ob = tkbn.KbNufft(im_size=self.im_size, grid_size=self.grid_size)

        self.per_scene = per_scene

        if apix_path is not None and os.path.exists(apix_path):
            self.apix_pd = pd.read_csv(apix_path)
            # process to a directory: pdb_id, apix
            self.apix_dir = {id: float(apix) for id, apix in zip(self.apix_pd['pdb_id'], self.apix_pd['apix']) if apix >= 2.0}
            print('load apix.csv')
        else:
            self.apix_dir = None

        if self.per_scene:
            image_dir = self.image_dirs[0]
            self.name = os.path.basename(image_dir)
            self.images = np.load(os.path.join(image_dir, 'particle_stack.npy')).astype(np.float32)
            self.poses = np.load(os.path.join(image_dir, 'poses.npy')).astype(np.float32)

            pdb_id = os.path.basename(image_dir)
            if self.apix_dir is not None and pdb_id in self.apix_dir:
                self.cached_apix = self.apix_dir[pdb_id]
                print(f'load apix for {pdb_id}, {self.cached_apix}')
            elif apix is not None:
                self.cached_apix = apix
            else:
                self.cached_apix = None
                
            if os.path.exists(os.path.join(image_dir, 'ctf.npy')):
                self.ctfs = np.load(os.path.join(image_dir, 'ctf.npy')).astype(np.float32)
                self.chis = compute_chi_from_drgn_ctfs(self.images[0].shape[-1], self.ctfs)
                D = self.ctfs[0, 0]
                Apix = self.ctfs[0, 1]
                self.apix = D / self.images.shape[-1] * Apix
                self.cached_apix = self.apix
                print(f'load ctf.npy, apix = {self.apix}')
            else:
                self.ctfs = None
                self.chis = None
            if os.path.exists(os.path.join(image_dir, 'translations.npy')):
                self.translations = np.load(os.path.join(image_dir, 'translations.npy')).astype(np.float32) # DRGN format (N, 2) [-1, 1] need to be multiplied by D
            elif os.path.exists(os.path.join(image_dir, 'trans.npy')):
                self.translations = np.load(os.path.join(image_dir, 'trans.npy')).astype(np.float32) / 128.0
                self.translations = self.translations[:, :2]
            else:
                self.translations = None

            if os.path.exists(os.path.join(image_dir, 'random_training_index.npy')):
                indices_to_be_excluded = np.load(os.path.join(image_dir, 'random_training_index.npy'))
                print(f"Excluding {len(indices_to_be_excluded)} images")
                self.images = np.delete(self.images, indices_to_be_excluded, axis=0)
                self.poses = np.delete(self.poses, indices_to_be_excluded, axis=0)
                if self.ctfs is not None:
                    self.ctfs = np.delete(self.ctfs, indices_to_be_excluded, axis=0)
                    self.chis = np.delete(self.chis, indices_to_be_excluded, axis=0)
                if self.translations is not None:
                    self.translations = np.delete(self.translations, indices_to_be_excluded, axis=0)
            
            if first_N is not None:
                self.images = self.images[:first_N]
                self.poses = self.poses[:first_N]
                if self.ctfs is not None:
                    self.ctfs = self.ctfs[:first_N]
                    self.chis = self.chis[:first_N]
                if self.translations is not None:
                    self.translations = self.translations[:first_N]
            
            if skip_first_N > 0:
                self.images = self.images[skip_first_N:]
                self.poses = self.poses[skip_first_N:]
                if self.ctfs is not None:
                    self.ctfs = self.ctfs[skip_first_N:]
                    self.chis = self.chis[skip_first_N:]
                if self.translations is not None:
                    self.translations = self.translations[skip_first_N:]

        else:
            self.translations = None

        # data augmentation
        self.transform = transform
        self.image_scalar = image_scalar

    def create_ktraj(self, nspokes=405, spokelength=128, sampling_scheme='uniform'):

        # sampling_scheme = ['golden_angle', 'uniform'][1]

        kx = np.zeros(shape=(spokelength, nspokes))
        ky = np.zeros(shape=(spokelength, nspokes))

        if sampling_scheme == 'golden_angle':
            # sample by the golden angle
            ga = np.deg2rad(180 / ((1 + np.sqrt(5)) / 2))
            ky[:, 0] = np.linspace(-np.pi, np.pi, spokelength)
            for i in range(1, nspokes):
                kx[:, i] = np.cos(ga) * kx[:, i - 1] - np.sin(ga) * ky[:, i - 1]
                ky[:, i] = np.sin(ga) * kx[:, i - 1] + np.cos(ga) * ky[:, i - 1]
        elif sampling_scheme == 'uniform':
            # sample uniformly
            angles = np.linspace(0, np.pi, nspokes)
            kx[:, 0] = np.linspace(-np.pi, np.pi, spokelength)
            for i in range(1, nspokes):
                angle = angles[i]
                kx[:, i] = np.cos(angle) * kx[:, 0]
                ky[:, i] = np.sin(angle) * kx[:, 0]

        ky = np.transpose(ky)
        kx = np.transpose(kx)

        ktraj = np.stack((ky.flatten(), kx.flatten()), axis=0)
        ktraj = torch.tensor(ktraj, dtype=torch.float32)
        return ktraj

    def __len__(self):
        # print(f'{self.per_scene = }')
        if self.per_scene:
            return self.poses.shape[0]
        return len(self.image_dirs)

    def estimate_normalization(self, n=1000):
        N = len(self.image_dirs)
        n = min(n, N)
        # indices = [i for i in range(0, N, N // n)]
        # mgs = self.src.images(indices)
        imgs = []
        image_dir = [self.image_dirs[idx] for idx in range(0, N, N // n)] # FIXME: what if the data is not IID??

        for dir in image_dir:
            img = np.load(os.path.join(dir, 'particle_stack.npy')).astype(np.float32)
            if len(img.shape) == 4:
                img = img[0, 0]
            elif len(img.shape) == 3:
                img = img[0]
            img = torch.from_numpy(img)
            imgs.append(ht2_center(img))

        imgs = torch.stack(imgs)

        # if self.invert_data:
        #     imgs *= -1

        imgs = symmetrize_ht(imgs)

        norm = (0, torch.std(imgs))
        print("Normalizing HT by {} +/- {}".format(*norm))
        return norm


    def _process(self, data, mode='ht'):

        orignal_shape = data.shape

        if len(orignal_shape) > 3:
            old = data.shape
            data = data.reshape(-1, old[-2], old[-1])
        elif len(orignal_shape) == 2:
            data = data[None, :, :]
            raise ValueError("Unsupported data input shape when processing images, should be (B, H, W)")

        B, H, W = data.shape
        # data = (data - data.min()) / (data.max() - data.min())
        data = torch.from_numpy(data)

        if mode == 'ht':
            data = ht2_center(data)
            data = (data - self.norm[0]) / self.norm[1]
            data = torch.abs(data)
            data = torch.log(data + 1)
        elif mode == 'unfft':
            data = data[:, None, :, :]
            data = self.nufft_ob(data, self.ktraj)
            data = np.log10(np.absolute(data))
        elif mode == 'none':
            return data.reshape(orignal_shape)
        elif mode == 'std':
            data = (data - data.mean()) / data.std()
            data = data / data.max()
        else:
            # normalize the data
            data = (data - data.min()) / (data.max() - data.min())

        if orignal_shape != data.shape:
            data = data.reshape(orignal_shape)

        return data


    def __getitem__(self, index):
        # set-up the rng
        if self.seed:  # reseed for each __getitem__
            self._rng = np.random.default_rng(seed=self.seed + index)
        elif not hasattr(self, '_rng'):
            seed = torch.initial_seed()  # this is different for each dataloader process
            self._rng = np.random.default_rng(seed=seed)

        # Load the multi-view images
        # image_dir = os.path.join(self.image_dirs[index], 'N10_snrNone') # TODO: remove subdir
        image_dir = self.image_dirs[0 if self.per_scene else index]
        pdb_id = os.path.basename(image_dir)
        apix = self.apix_dir.get(pdb_id, None) if self.apix_dir else None

        if self.per_scene:
            images = self.images
            poses = self.poses
            chis = self.chis if self.chis is not None else None
        else:
            images = np.load(os.path.join(image_dir, 'particle_stack.npy')).astype(np.float32)
            poses = np.load(os.path.join(image_dir, 'poses.npy')).astype(np.float32)
            chis = None
        
        if self.translations is not None:
            translations = self.translations
        elif os.path.exists(os.path.join(image_dir, 'translations.npy')):
            translations = np.load(os.path.join(image_dir, 'translations.npy')).astype(np.float32) # DRGN format (N, 2) [-1, 1] need to be multiplied by D
        else:
            translations = None

        # transpose poses for 3D gaussian projections, no actual effects to fourier reconstruction
        # col1 = poses[...,:1]
        # col2 = poses[...,1:2]
        # col3 = poses[...,2:]
        # poses = np.concatenate([col3, col2, col1], axis=-1)

        # check if the dataset has already save multi-view image structure
        if len(images.shape) == 3:
            N, H, W = images.shape
            # randomly select two image indices in images
            if self.min_angle or self.max_angle:
                valid_path = os.path.join(image_dir, f'view_{self.num_views}_min_{self.min_angle}_max_{self.max_angle}.npy')
                if os.path.exists(valid_path):
                    valid_indices = np.load(valid_path)
                else:
                    # calculate the valid combination

                    indices = [i for i in range(N)]
                    combinations = list(itertools.combinations(indices, self.num_views))

                    valid_indices = []
                    # tranverse combos
                    for combo in combinations:
                        temp = poses[list(combo)]
                        if valid_rotation(temp, self.min_angle, self.max_angle):
                            valid_indices.append(combo)

                    valid_indices = np.array(valid_indices)
                    np.save(valid_path, valid_indices)

                image_indices = np.random.choice(len(valid_indices), self.mini_batch_size, replace=False) # (mini_batch_size, num_views)
                image_indices = valid_indices[image_indices] # (N, M)
                image_indices = image_indices.transpose(1, 0).reshape(-1) # (N, M) -> (M, N)
            else:
                final_indices = []
                for i in range(self.mini_batch_size):
                    image_indices = np.random.choice(images.shape[0], self.num_views, replace=False)
                    final_indices.append(image_indices)
                
                image_indices = np.array(final_indices).reshape(-1)

            mini_batch_images = images[image_indices]
            mini_batch_poses = poses[image_indices]
            mini_batch_chis = chis[image_indices] if chis is not None else None
            mini_batch_translations = translations[image_indices] if translations is not None else None

        elif len(images.shape) == 4:
            N, M, H, W = images.shape
            assert M >= self.num_views, ValueError("No enough view number in loaded data.")

            if M > self.num_views:
                images = images[:, :self.num_views, :, :]
                poses = poses[:, :self.num_views, :, :]

            # M is the number of multi-view images
            # randomly select two image indices in images
            image_indices = np.random.choice(images.shape[0], self.mini_batch_size, replace=False)
            mini_batch_images = images[image_indices].transpose(1, 0, 2, 3).reshape(self.mini_batch_size * self.num_views, H, W) # (N, M) -> (M, N)
            mini_batch_poses = poses[image_indices].transpose(1, 0, 2, 3).reshape(self.mini_batch_size * self.num_views, 3, 3)

        mini_batch_images = self._process(mini_batch_images, self.mode)
        mini_batch_images = mini_batch_images * self.image_scalar
        # split the images into multi-view
        if mini_batch_translations is not None:
            mini_batch_trans = np.concatenate([mini_batch_translations, np.zeros_like(mini_batch_translations[...,:1])], axis=1).astype(np.float32).reshape(-1, 3)
            mini_batch_trans = np.split(mini_batch_trans, self.num_views, axis=0) if translations is not None else None
            mini_batch_translations = np.split(mini_batch_translations, self.num_views, axis=0) if translations is not None else None
        
        mini_batch_images = np.split(mini_batch_images, self.num_views, axis=0)
        mini_batch_poses = np.split(mini_batch_poses, self.num_views, axis=0)
        mini_batch_chis = np.split(mini_batch_chis, self.num_views, axis=0) if chis is not None else None

        if mini_batch_chis is not None:
            views = [{"img": mini_batch_images[i], "pose": mini_batch_poses[i], "chi": mini_batch_chis[i]} for i in range(self.num_views)]
        else:
            views = [{"img": mini_batch_images[i], "pose": mini_batch_poses[i]} for i in range(self.num_views)]

        if mini_batch_translations is not None:
            for i in range(self.num_views):
                views[i]['translation'] = mini_batch_translations[i]
                views[i]['trans'] = mini_batch_trans[i]

        # Apply transformations if specified
        if self.transform:
            views = self.transform(views, apix=apix)

        # Return the multiview images with poses
        return views

    def get_num_of_images(self, pid):
        image_dir = self.image_dirs[pid]
        poses = np.load(os.path.join(image_dir, 'poses.npy')).astype(np.float32)
        return poses.shape[0]

    def get_views(self, pid, vids):
        # put images and poses into memory as sometimes they are pretty heavy to be loaded every time.
        if self.per_scene:
            images = self.images
            poses = self.poses
            if self.chis is not None:
                chis = self.chis
            else:
                chis = None
            if self.translations is not None:
                translations = self.translations
            else:
                translations = None
        elif pid not in self.cached_images:
            image_dir = self.image_dirs[pid]
            images = np.load(os.path.join(image_dir, 'particle_stack.npy')).astype(np.float32)
            poses = np.load(os.path.join(image_dir, 'poses.npy')).astype(np.float32)
            self.cached_images[pid] = images
            self.cached_poses[pid] = poses
        else:
            images = self.cached_images[pid]
            poses = self.cached_poses[pid]

        views = []
        image_list = []
        pose_list = []
        chi_list = []
        translation_list = []

        for vid in vids:
            if len(images.shape) == 3:
                image = images[vid][None, :, :]
                pose = poses[vid]
                chi = chis[vid] if chis is not None else None
                translation = translations[vid] if translations is not None else None
            elif len(images.shape) == 4:
                image = images[vid, 0][None, :, :]
                pose = poses[vid, 0]
            else:
                raise ValueError("Unsupported shape of input images")
            image_list.append(image)
            pose_list.append(pose)
            chi_list.append(chi)
            translation_list.append(translation)

        image_list = self._process(np.stack(image_list, axis=0), self.mode)
        image_list = image_list * self.image_scalar
        pose_list = np.stack(pose_list, axis=0)
        chi_list = np.stack(chi_list, axis=0) if chis is not None else None
        translation_list = np.stack(translation_list, axis=0) if translations is not None else None
        for i in range(len(image_list)):
            image = image_list[i]
            pose = pose_list[i]
            view = {"img": image, "pose": pose, "id": vids[i]}
            if chis is not None:
                view['chi'] = chi_list[i]
            if translations is not None:
                view['translation'] = translation_list[i]
            views.append(view)

        # Apply transformations if specified
        if self.transform:
            views = self.transform(views, apix=self.cached_apix)
        
        return views


    def get_pair(self, pid, vid1, vid2):
        # put images and poses into memory as sometimes they are pretty heavy to be loaded every time.
        if pid not in self.cached_images:
            image_dir = self.image_dirs[pid]
            images = np.load(os.path.join(image_dir, 'particle_stack.npy')).astype(np.float32)
            poses = np.load(os.path.join(image_dir, 'poses.npy')).astype(np.float32)
            self.cached_images[pid] = images
            self.cached_poses[pid] = poses
        else:
            images = self.cached_images[pid]
            poses = self.cached_poses[pid]


        if len(images.shape) == 3:
            image1 = images[vid1][None, :, :]
            image2 = images[vid2][None, :, :]
        elif len(images.shape) == 4:
            image1 = images[vid1, 0][None, :, :]
            image2 = images[vid2, 0][None, :, :]
        else:
            raise ValueError("Unsupported shape of input images")

        image1 = self._process(image1, self.mode)
        image2 = self._process(image2, self.mode)

        pose1 = poses[vid1]
        pose2 = poses[vid2]

        view1 = {"img": image1, "pose": pose1}
        view2 = {"img": image2, "pose": pose2}

        return view1, view2
