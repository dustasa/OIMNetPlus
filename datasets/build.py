import torch

from utils.transforms import build_transforms
from utils.utils import create_small_table

from datasets.cuhk_sysu import CUHKSYSU
from datasets.prw import PRW
from datasets.my_dataset_prwvoc import PRW_VOCInstances
from datasets.prw_voc import PRW_VOC
from defaults import get_default_cfg


def print_statistics(dataset):
    """
    Print dataset statistics.
    """
    num_imgs = len(dataset.annotations)
    num_boxes = 0
    pid_set = set()
    for anno in dataset.annotations:
        num_boxes += anno["boxes"].shape[0]
        for pid in anno["pids"]:
            pid_set.add(pid)
    statistics = {
        "dataset": dataset.name,
        "split": dataset.split,
        "num_images": num_imgs,
        "num_boxes": num_boxes,
    }
    if dataset.name != "CUHK-SYSU" or dataset.split != "query":
        pid_list = sorted(list(pid_set))
        if dataset.split == "query":
            num_pids, min_pid, max_pid = len(pid_list), min(pid_list), max(pid_list)
            statistics.update(
                {
                    "num_labeled_pids": num_pids,
                    "min_labeled_pid": int(min_pid),
                    "max_labeled_pid": int(max_pid),
                }
            )
        else:
            unlabeled_pid = pid_list[-1]
            pid_list = pid_list[:-1]  # remove unlabeled pid
            num_pids, min_pid, max_pid = len(pid_list), min(pid_list), max(pid_list)
            statistics.update(
                {
                    "num_labeled_pids": num_pids,
                    "min_labeled_pid": int(min_pid),
                    "max_labeled_pid": int(max_pid),
                    "unlabeled_pid": int(unlabeled_pid),
                }
            )
    print(f"=> {dataset.name}-{dataset.split} loaded:\n" + create_small_table(statistics))


def build_dataset(dataset_name, root, transforms, split, verbose=True):
    if dataset_name == "CUHK-SYSU":
        dataset = CUHKSYSU(root, transforms, split)
    elif dataset_name == "PRW":
        dataset = PRW(root, transforms, split)
    elif dataset_name == "PRW_VOC":
        dataset = PRW_VOC(root, transforms, split)
    else:
        raise NotImplementedError(f"Unknow dataset: {dataset_name}")
    if verbose:
        print_statistics(dataset)
    d1 = dataset[0]
    print(f'build dataset [67]: {d1}')
    return dataset


def collate_fn(batch):
    return tuple(zip(*batch))


def build_train_loader(cfg):
    transforms = build_transforms(is_train=True)
    dataset = build_dataset(cfg.INPUT.DATASET, cfg.INPUT.DATA_ROOT, transforms, "train")
    d1 = dataset[0]
    print(f'build_train_loader [79]: {d1}')
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.INPUT.BATCH_SIZE_TRAIN,
        shuffle=True,
        num_workers=cfg.INPUT.NUM_WORKERS_TRAIN,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )


def build_test_loader(cfg):
    transforms = build_transforms(is_train=False)
    gallery_set = build_dataset(cfg.INPUT.DATASET, cfg.INPUT.DATA_ROOT, transforms, "gallery")
    d1 = gallery_set[0]
    print(f'gallery_set [95]: {d1}')
    query_set = build_dataset(cfg.INPUT.DATASET, cfg.INPUT.DATA_ROOT, transforms, "query")
    d2 = query_set[0]
    print(f'query_set [98]: {d2}')
    gallery_loader = torch.utils.data.DataLoader(
        gallery_set,
        batch_size=cfg.INPUT.BATCH_SIZE_TEST,
        shuffle=False,
        num_workers=cfg.INPUT.NUM_WORKERS_TEST,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    query_loader = torch.utils.data.DataLoader(
        query_set,
        batch_size=cfg.INPUT.BATCH_SIZE_TEST,
        shuffle=False,
        num_workers=cfg.INPUT.NUM_WORKERS_TEST,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    return gallery_loader, query_loader


def main(cfg):
    train_loader = build_train_loader(cfg)
    print(len(train_loader))
    # train1 = train_loader[0]
    # print(train1)
    gallery_loader, query_loader = build_test_loader(cfg)
    print(len(gallery_loader))
    print(len(query_loader))


if __name__ == '__main__':
    cfg = get_default_cfg()
    main(cfg)
