import os
import sys

import argparse

import h5py
import torch.utils.data
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm


class DatasetImages(torch.utils.data.Dataset):
    """ Dataset for MSCOCO images located in a folder on the filesystem """

    def __init__(self, path, transform=None):
        super().__init__()
        self.path = path

        if "clevr" in path:
            self.id_to_filename = self._find_images_clevr()
        elif "cats" in path:
            self.id_to_filename = self._find_images_cats()
        elif "faces" in path:
            self.id_to_filename = self._find_images_faces()
        elif "wflw" in path:
            self.id_to_filename = self._find_images_wflw()

        self.sorted_ids = sorted(
            self.id_to_filename.keys()
        )  # used for deterministic iteration order
        print(f"found {len(self)} images in {self.path}")
        self.transform = transform

    def _find_images_clevr(self):
        id_to_filename = {}
        for filename in os.listdir(self.path):
            if not filename.endswith(".png"):
                continue
            id_and_extension = filename.split("_")[-1]
            id = int(id_and_extension.split(".")[0])
            id_to_filename[id] = filename
        return id_to_filename

    def _find_images_cats(self):
        id_to_filename = {}
        for filename in os.listdir(self.path):
            if not filename.endswith(".jpg"):
                continue
            id = filename.split(".")[0]
            id = int(id.replace("_", ""))
            if id in id_to_filename:
                print("non_unique IDS found")
            id_to_filename[id] = filename
        return id_to_filename

    def _find_images_wflw(self):
        id_to_filename = {}
        for filename in os.listdir(self.path):
            if not filename.endswith(".jpg"):
                continue
            id = filename.split(".")[0]
            id = int(id.replace("_", ""))
            if id in id_to_filename:
                print("non_unique IDS found")
            id_to_filename[id] = filename
        return id_to_filename


    def _find_images_faces(self):
        return {int(f.split('.')[0]): f for f in os.listdir(self.path) if f.endswith(".png")}

    def __getitem__(self, item):
        id = self.sorted_ids[item]
        path = os.path.join(self.path, self.id_to_filename[id])
        img = Image.open(path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)
        return id, img

    def __len__(self):
        return len(self.sorted_ids)

def create_coco_loader(path):
    transform = transforms.Compose(
        [transforms.Resize((128, 128)), transforms.ToTensor()]
    )
    dataset = DatasetImages(path, transform=transform)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=64, num_workers=12, shuffle=False, pin_memory=True
    )
    return data_loader


def main(dataset):
    base_path = os.path.join(os.getcwd(), dataset)
    for split_name in ["train", "val"]:
        path = os.path.join(base_path, "images", split_name)
        loader = create_coco_loader(path)
        images_shape = (len(loader.dataset), 3, 128, 128)

        h5_path = os.path.join(base_path, f"{split_name}-images.h5")
        with h5py.File(h5_path, "w", libver="latest") as fd:
            images = fd.create_dataset("images", 
                                       shape=images_shape, 
                                       dtype="float32")
            image_ids = fd.create_dataset("image_ids",
                                          shape=(len(loader.dataset), ), 
                                          dtype="int32")

            i = 0
            for ids, imgs in tqdm(loader):
                j = i + imgs.size(0)
                images[i:j, :, :] = imgs.numpy()
                image_ids[i:j] = ids.numpy().astype("int32")
                i = j


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", default="cats", help="Dataset to process (cats/clevr)", choices=["cats", "clevr", "faces"]
    )

    args = parser.parse_args()
    main(args.dataset)
