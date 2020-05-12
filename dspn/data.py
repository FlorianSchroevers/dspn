import os
import math
import random
import json

import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as T
import h5py
import numpy as np

from PIL import Image


def get_loader(dataset, batch_size, num_workers=1, shuffle=True):
    return torch.utils.data.DataLoader(
        dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=True,
    )


class MNISTSet(torch.utils.data.Dataset):
    def __init__(self, threshold=0.0, train=True, root="mnist", full=False):
        self.train = train
        self.root = root
        self.threshold = threshold
        self.full = full
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        mnist = torchvision.datasets.MNIST(
            train=train, transform=transform, download=True, root=root
        )
        self.data = self.cache(mnist)
        self.max = 342

    def cache(self, dataset):
        cache_path = os.path.join(self.root, f"mnist_{self.train}_{self.threshold}.pth")
        if os.path.exists(cache_path):
            return torch.load(cache_path)

        print("Processing dataset...")
        data = []
        for datapoint in dataset:
            img, label = datapoint
            point_set, cardinality = self.image_to_set(img)
            data.append((point_set, label, cardinality))
        torch.save(data, cache_path)
        print("Done!")
        return data

    def image_to_set(self, img):
        idx = (img.squeeze(0) > self.threshold).nonzero().transpose(0, 1)
        cardinality = idx.size(1)
        return idx, cardinality

    def __getitem__(self, item):
        s, l, c = self.data[item]
        # make sure set is shuffled
        s = s[:, torch.randperm(c)]
        # pad to fixed size
        padding_size = self.max - s.size(1)
        s = torch.cat([s.float(), torch.zeros(2, padding_size)], dim=1)
        # put in range [0, 1]
        s = s / 27
        # mask of which elements are valid,not padding
        mask = torch.zeros(self.max)
        mask[:c].fill_(1)
        return l, s, mask

    def __len__(self):
        if self.train or self.full:
            return len(self.data)
        else:
            return len(self.data) // 10


CLASSES = {
    "material": ["rubber", "metal"],
    "color": ["cyan", "blue", "yellow", "purple", "red", "green", "gray", "brown"],
    "shape": ["sphere", "cube", "cylinder"],
    "size": ["large", "small"],
}

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, split, max_objects, full=False):
        assert split in {
            "train",
            "val",
            "test",
        }  # note: test isn't very useful since it doesn't have ground-truth scene information
        self.base_path = base_path
        self.split = split
        self.max_objects = max_objects
        self.full = full # Use full validation set?

        # the following only works if the index is not enumerated (string or
        # random numbers), overwrite if necesarry
        with self.img_db() as db:
            ids = db["image_ids"]
            self.image_id_to_index = {id: i for i, id in enumerate(ids)}

        self.image_db = self.img_db()

    @property
    def images_folder(self):
        return os.path.join(self.base_path, "images", self.split)

    def img_db(self):
        path = os.path.join(self.base_path, f"{self.split}-images.h5")
        return h5py.File(path, "r")

    def __getitem__(self, item):
        img_id = self.image_db["image_ids"][item]
        img_idx = self.image_id_to_index[img_id]

        targets, size = self.targets[img_idx]
        image = self.image_db["images"][item]
        return image, targets, size

    def __len__(self):
        if self.split == "train" or self.full:
            return len(self.image_db["images"])
        else:
            return len(self.image_db["images"]) // 10


class CLEVR(ImageDataset):
    def __init__(self, base_path, split, box=False, full=False):
        super().__init__(base_path, split, 10, full)

        self.box = box

        with open(self.scenes_path) as fd:
            scenes = json.load(fd)["scenes"]
        self.img_ids, self.targets = self.prepare_scenes(scenes)

    def display(item):
        img, targets, size = self.__getitem__[item]

        plt.scatter(targets[0, :]*128, targets[1, :]*128, c='r')

        plt.imshow(np.transpose(img, (1, 2, 0)))
        plt.show()

    def object_to_fv(self, obj):
        coords = [p / 3 for p in obj["3d_coords"]]
        one_hot = lambda key: [obj[key] == x for x in CLASSES[key]]
        material = one_hot("material")
        color = one_hot("color")
        shape = one_hot("shape")
        size = one_hot("size")
        assert sum(material) == 1
        assert sum(color) == 1
        assert sum(shape) == 1
        assert sum(size) == 1
        # concatenate all the classes
        return coords + material + color + shape + size

    def prepare_scenes(self, scenes_json):
        img_ids = []
        targets = []
        for scene in scenes_json:
            img_idx = scene["image_index"]
            # different objects depending on bbox version or attribute version of CLEVR sets
            if self.box:
                objects = self.extract_bounding_boxes(scene)
                objects = torch.FloatTensor(objects)
            else:
                objects = [self.object_to_fv(obj) for obj in scene["objects"]]
                objects = torch.FloatTensor(objects).transpose(0, 1)
            num_objects = objects.size(1)
            # pad with 0s
            if num_objects < self.max_objects:
                objects = torch.cat(
                    [
                        objects,
                        torch.zeros(objects.size(0), self.max_objects - num_objects),
                    ],
                    dim=1,
                )
            # fill in masks
            mask = torch.zeros(self.max_objects)
            mask[:num_objects] = 1

            img_ids.append(img_idx)
            targets.append((objects, mask))
        return img_ids, targets

    def extract_bounding_boxes(self, scene):
        """
        Code used for 'Object-based Reasoning in VQA' to generate bboxes
        https://arxiv.org/abs/1801.09718
        https://github.com/larchen/clevr-vqa/blob/master/bounding_box.py#L51-L107
        """
        objs = scene["objects"]
        rotation = scene["directions"]["right"]

        num_boxes = len(objs)

        boxes = np.zeros((1, num_boxes, 4))

        xmin = []
        ymin = []
        xmax = []
        ymax = []
        classes = []
        classes_text = []

        for i, obj in enumerate(objs):
            [x, y, z] = obj["pixel_coords"]

            [x1, y1, z1] = obj["3d_coords"]

            cos_theta, sin_theta, _ = rotation

            x1 = x1 * cos_theta + y1 * sin_theta
            y1 = x1 * -sin_theta + y1 * cos_theta

            height_d = 6.9 * z1 * (15 - y1) / 2.0
            height_u = height_d
            width_l = height_d
            width_r = height_d

            if obj["shape"] == "cylinder":
                d = 9.4 + y1
                h = 6.4
                s = z1

                height_u *= (s * (h / d + 1)) / ((s * (h / d + 1)) - (s * (h - s) / d))
                height_d = height_u * (h - s + d) / (h + s + d)

                width_l *= 11 / (10 + y1)
                width_r = width_l

            if obj["shape"] == "cube":
                height_u *= 1.3 * 10 / (10 + y1)
                height_d = height_u
                width_l = height_u
                width_r = height_u

            obj_name = (
                obj["size"]
                + " "
                + obj["color"]
                + " "
                + obj["material"]
                + " "
                + obj["shape"]
            )
            ymin.append((y - height_d) / 320.0)
            ymax.append((y + height_u) / 320.0)
            xmin.append((x - width_l) / 480.0)
            xmax.append((x + width_r) / 480.0)

        return xmin, ymin, xmax, ymax

    @property
    def scenes_path(self):
        if self.split == "test":
            raise ValueError("Scenes are not available for test")
        return os.path.join(
            self.base_path, "scenes", f"CLEVR_{self.split}_scenes.json"
        )

class Cats(ImageDataset):
    def __init__(self, base_path, split, full=False):
        super().__init__(base_path, split, 10, full)

        ids, self.targets = self.prepare_keypoints()

        self.image_id_to_index = {img_id: i for i, img_id in enumerate(ids)}

    def prepare_keypoints(self):
        targets = []
        ids = []

        # for img_id in self.image_db["image_ids"]:
        for fname in os.listdir(self.images_folder):
            # fname = str(img_id)[:9] + "_" + str(img_id)[10:] + ".jpg.cat"
            # print(img_id)

            if not fname.endswith(".cat"):
                continue

            with open(os.path.join(self.images_folder, fname), "r") as f:
                # read an image to get its dimensions. In the cats dataset
                # the dimensions are not uniform. 
                # this does not load the entire image into memory
                img_path = os.path.join(self.images_folder, fname[:-4])
                im_x, im_y = Image.open(img_path).size

                # there are 19 numbers in the .cat file, the first of which 
                # denotes the number of keypoints. the coordinates of 
                # keypoints are given by the remaining 18 numbers, 
                # where consecutive numbers denote an x, y coordinate.
                # we can remove the 0th element and there is a trailing 
                # whitespace we can remove
                nums = f.read().split(" ")[1:-1]
                coords_flat = np.array([int(n) for n in nums])

                # get into right shape and adjust for image size
                coords = coords_flat.reshape(-1, 2).T / [[im_x], [im_y]]
                n_objects = coords.shape[1]
                
                if n_objects > self.max_objects:
                    raise IndexError("Number of objects exceeds estimated"
                                     "max number of object")
                
                # overlay objects on zero array holding up to max_objects
                objects = np.zeros(shape=(2, self.max_objects))
                objects[:, :n_objects] = coords
            
            objects = torch.FloatTensor(objects)

            # fill in masks
            mask = torch.zeros(self.max_objects)
            mask[:n_objects] = 1

            targets.append((objects, mask))
            ids.append(int(fname.split(".")[0].replace("_", "")))

        return ids, targets


class Faces(ImageDataset):
    def __init__(self, base_path, split, full=False):
        super().__init__(base_path, split, 10, full)

        self.targets = self.prepare_keypoints()

    def prepare_keypoints(self):

        # there are multiple keypoints per major feature (left eye, right eye.
        # nose and mouth), so we want to calculate the mean of those keypoints
        # per major feature, these are the indices for those features
        left_eye_x_indices = [0, 4, 6, 12, 14]
        left_eye_y_indices = [1, 5, 7, 13, 15]
        right_eye_x_indices = [2, 8, 10, 16, 18]
        right_eye_y_indices = [3, 9, 11, 17, 19]
        nose__x_indices = [20]
        nose__y_indices = [21]
        mouth_x_indices = [22, 24, 26, 28]
        mouth_y_indices = [23, 25, 27, 29]

        # a dict mapping the final position in the array to one of the major facial keypoints
        features = {
            0: (left_eye_x_indices, left_eye_y_indices),
            1: (right_eye_x_indices, right_eye_y_indices),
            2: (nose__x_indices, nose__y_indices),
            3: (mouth_x_indices, mouth_y_indices)
        }
        
        keypoints_array = np.genfromtxt(
            os.path.join(self.base_path, "facial_keypoints.csv"),
            delimiter=',',
            # usecols = keypoints_to_keep,
            missing_values = 0
        )

        keypoints_array = np.nan_to_num(keypoints_array, copy=False)

        targets = []
        for img_keypoints in keypoints_array:
            # four keypoints, x and y components
            coords = np.empty((2, 4))

            # calculate means without zeros, and normalize by dividing by 
            # image size (96x96)
            for i in features:
                for dim in [0, 1]:
                    total = img_keypoints[features[i][dim]].sum()
                    count = (img_keypoints[features[i][dim]] != 0).sum()
                    # in case the count is zero, the value should just be zero
                    if count == 0:
                        coords[dim, i] = 0
                    else:
                        coords[dim, i] = (total / count) / 96


            # overlay objects on zero array holding up to max_objects
            objects = np.zeros(shape=(2, self.max_objects))
            objects[:, :coords.shape[1]] = coords
            
            objects = torch.FloatTensor(objects)

            # fill in masks
            mask = torch.zeros(self.max_objects)
            mask[:coords.shape[1]] = 1

            targets.append((objects, mask))


        return targets

class MergedDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.targets = []
        self.datasets = datasets

        for i in range(1, len(datasets)):
            assert datasets[i].max_objects == datasets[i - 1].max_objects
            
    def __getitem__(self, item):
        total = 0
        for dataset in self.datasets:
            if item < len(dataset) + total:
                return dataset[item - total - 1]
            else:
                total += len(dataset)

        raise IndexError()

    def __len__(self):
        return sum([len(ds) for ds in self.datasets])

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    faces = Faces("faces", "train")
    print("Loaded faces")
    cats = Cats("cats", "train")
    print("Loaded cats")

    merged = MergedDataset(faces, cats)
    while True:
        img, keypoints, size = merged[np.random.randint(len(merged))]

        fig = plt.figure()
        plt.scatter(keypoints[0, :]*128, keypoints[1, :]*128, c='r')

        plt.imshow(np.transpose(img, (1, 2, 0)))
        plt.show()

