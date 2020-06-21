import os
import argparse
from datetime import datetime
import time

import torch
import torch.nn.functional as F
import torch.multiprocessing as mp

import numpy as np
from tqdm import tqdm
import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
# import tkinter
# matplotlib.use('TkAgg')

from tensorboardX import SummaryWriter

import data
import track
import model
import utils

import pandas as pd

matplotlib.use("Qt5Agg")

def main():
    global net
    global test_loader
    global scatter
    parser = argparse.ArgumentParser()
    # generic params
    parser.add_argument(
        "--name",
        default=datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
        help="Name to store the log file as",
    )
    parser.add_argument("--resume", help="Path to log file to resume from")

    parser.add_argument("--encoder", default="FSEncoder", help="Encoder")
    parser.add_argument("--decoder", default="DSPN", help="Decoder")
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs to train with"
    )
    parser.add_argument(
        "--latent", type=int, default=32, help="Dimensionality of latent space"
    )
    parser.add_argument(
        "--dim", type=int, default=64, help="Dimensionality of hidden layers"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-2, help="Outer learning rate of model"
    )
    parser.add_argument(
        "--batch-size", type=int, default=12, help="Batch size to train with"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of threads for data loader"
    )
    parser.add_argument(
        "--dataset",
        choices=[
            "mnist", "clevr-box", "clevr-state", "cats", "faces", "merged", "wflw"
        ],
        help="Which dataset to use",
    )
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        help="Run on CPU instead of GPU (not recommended)",
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Only run training, no evaluation"
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only run evaluation, no training"
    )
    parser.add_argument(
        "--multi-gpu",
        action="store_true",
        help="Use multiple GPUs"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Plot generated samples in Tensorboard"
    )
    parser.add_argument(
        "--show-skip",
        type=int,
        default=1,
        help="Number of epochs to skip before exporting to Tensorboard"
    )

    parser.add_argument(
        "--infer-name",
        action="store_true",
        help="Automatically name run based on dataset/run number"
    )

    parser.add_argument("--supervised", action="store_true", help="")
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Use baseline model"
    )

    parser.add_argument(
        "--export-dir", type=str, help="Directory to output samples to")
    parser.add_argument(
        "--export-n",
        type=int,
        default=10 ** 9,
        help="How many samples to output"
    )
    parser.add_argument(
        "--export-progress",
        action="store_true",
        help="Output intermediate set predictions for DSPN?",
    )
    parser.add_argument(
        "--full-eval",
        action="store_true",
        help="Use full evaluation set (default: 1/10 of evaluation data)",
        # don't need full evaluation when training to save some time
    )
    parser.add_argument(
        "--mask-feature",
        action="store_true",
        help="Treat mask as a feature to compute loss with",
    )
    parser.add_argument(
        "--inner-lr",
        type=float,
        default=800,
        help="Learning rate of DSPN inner optimisation",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=10,
        help="How many DSPN inner optimisation iteration to take",
    )
    parser.add_argument(
        "--huber-repr",
        type=float,
        default=1,
        help="Scaling of repr loss term for DSPN supervised learning",
    )
    parser.add_argument(
        "--loss",
        choices=["hungarian", "chamfer", "emd"],
        default="emd",
        help="Type of loss used",
    )
    parser.add_argument(
        "--export-csv",
        action="store_true",
        help="Only perform predictions, don't evaluate in any way"
    )

    args = parser.parse_args()

    if args.infer_name:
        if args.baseline:
            prefix = "base"
        else:
            prefix = "dspn"

        used_nums = []

        if not os.path.exists("runs"):
            os.makedirs("runs")

        runs = os.listdir("runs")
        for run in runs:
            if args.dataset in run:
                used_nums.append(int(run.split("-")[-1]))

        num = 1
        while num in used_nums:
            num += 1
        name = f"{prefix}-{args.dataset}-{num}"
    else:
        name = args.name

    print(f"Saving run to runs/{name}")
    train_writer = SummaryWriter(f"runs/{name}", purge_step=0)

    net = model.build_net(args)

    if not args.no_cuda:
        net = net.cuda()

    if args.multi_gpu:
        net = torch.nn.DataParallel(net)

    optimizer = torch.optim.Adam(
        [p for p in net.parameters() if p.requires_grad], lr=args.lr
    )

    print("Building dataloader")
    if args.dataset == "mnist":
        dataset_train = data.MNISTSet(train=True, full=args.full_eval)
        dataset_test = data.MNISTSet(train=False, full=args.full_eval)
    elif args.dataset in ["clevr-box", "clevr-state"]:
        dataset_train = data.CLEVR(
            "clevr",
            "train",
            box=args.dataset == "clevr-box",
            full=args.full_eval
        )

        dataset_test = data.CLEVR(
            "clevr",
            "val",
            box=args.dataset == "clevr-box",
            full=args.full_eval
        )
    elif args.dataset == "cats":
        dataset_train = data.Cats("cats", "train", 9, full=args.full_eval)
        dataset_test = data.Cats("cats", "val", 9, full=args.full_eval)
    elif args.dataset == "faces":
        dataset_train = data.Faces("faces", "train", 4, full=args.full_eval)
        dataset_test = data.Faces("faces", "val", 4, full=args.full_eval)
    elif args.dataset == "wflw":
        dataset_train = data.WFLW("wflw", "train", 7, full=args.full_eval)
        dataset_test = data.WFLW("wflw", "test", 7, full=args.full_eval)
    elif args.dataset == "merged":
        # merged cats and human faces
        dataset_train_cats = data.Cats("cats", "train", 9, full=args.full_eval)
        dataset_train_faces = data.WFLW("wflw", "train", 9, full=args.full_eval)

        dataset_test_cats = data.Cats("cats", "val", 9, full=args.full_eval)
        dataset_test_faces = data.WFLW("wflw", "test", 9, full=args.full_eval)

        dataset_train = data.MergedDataset(
            dataset_train_cats,
            dataset_train_faces
        )

        dataset_test = data.MergedDataset(
            dataset_test_cats,
            dataset_test_faces
        )

    if not args.eval_only:
        train_loader = data.get_loader(
            dataset_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )

    if not args.train_only:
        test_loader = data.get_loader(
            dataset_test,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False
        )

    tracker = track.Tracker(
        train_mae=track.ExpMean(),
        train_last=track.ExpMean(),
        train_loss=track.ExpMean(),
        test_mae=track.Mean(),
        test_last=track.Mean(),
        test_loss=track.Mean(),
    )

    if args.resume:
        log = torch.load(args.resume)
        weights = log["weights"]
        n = net
        if args.multi_gpu:
            n = n.module
        n.load_state_dict(weights, strict=True)


    if args.export_csv:
        predictions = {}

    def run(net, loader, optimizer, train=False, epoch=0, pool=None):
        writer = train_writer
        if train:
            net.train()
            prefix = "train"
            torch.set_grad_enabled(True)
        else:
            net.eval()
            prefix = "test"
            torch.set_grad_enabled(False)

        if args.export_dir:
            true_export = []
            pred_export = []

        iters_per_epoch = len(loader)
        loader = tqdm(
            loader,
            ncols=0,
            desc="{1} E{0:02d}".format(epoch, "train" if train else "test "),
        )

        for i, sample in enumerate(loader, start=epoch * iters_per_epoch):
            # input is either a set or an image
            input, target_set, target_mask = map(lambda x: x.cuda(), sample)

            # forward evaluation through the network
            (progress, masks, evals, gradn), (y_enc, y_label) = net(
                input, target_set, target_mask
            )

            progress_only = progress

            # if using mask as feature, concat mask feature into progress
            if args.mask_feature:
                target_set = torch.cat(
                    [target_set, target_mask.unsqueeze(dim=1)], dim=1
                )
                progress = [
                    torch.cat([p, m.unsqueeze(dim=1)], dim=1)
                    for p, m in zip(progress, masks)
                ]

            if args.loss == "chamfer":
                # dim 0 is over the inner iteration steps
                # target set is broadcasted over dim 0
                set_loss = utils.chamfer_loss(
                    torch.stack(progress), target_set.unsqueeze(0)
                )
            elif args.loss == "hungarian":
                set_loss = utils.hungarian_loss(
                    progress[-1], target_set, thread_pool=pool
                ).unsqueeze(0)
            elif args.loss == "emd":
                set_loss = utils.emd(progress[-1], target_set).unsqueeze(0)

            # Only use representation loss with DSPN and when doing general
            # supervised prediction, not when auto-encoding
            if args.supervised and not args.baseline:
                repr_loss = args.huber_repr * F.smooth_l1_loss(y_enc, y_label)
                loss = set_loss.mean() + repr_loss.mean()
            else:
                loss = set_loss.mean()

            # restore progress variable to not contain masks for correct
            # exporting
            progress = progress_only

            # Outer optim step
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Tensorboard tracking of metrics for debugging
            tracked_last = tracker.update(
                f"{prefix}_last", set_loss[-1].item()
            )
            tracked_loss = tracker.update(f"{prefix}_loss", loss.item())
            if train:
                writer.add_scalar(
                    "metric/set-loss",
                    loss.item(),
                    global_step=i
                )

                writer.add_scalar(
                    "metric/set-last",
                    set_loss[-1].mean().item(),
                    global_step=i
                )

                if not args.baseline:
                    writer.add_scalar(
                        "metric/eval-first",
                        evals[0].mean().item(),
                        global_step=i
                    )

                    writer.add_scalar(
                        "metric/eval-last",
                        evals[-1].mean().item(),
                        global_step=i
                    )

                    writer.add_scalar(
                        "metric/max-inner-grad-norm",
                        max(g.item() for g in gradn),
                        global_step=i
                    )

                    writer.add_scalar(
                        "metric/mean-inner-grad-norm",
                        sum(g.item() for g in gradn)/len(gradn),
                        global_step=i
                    )

                    if args.supervised:
                        writer.add_scalar(
                            "metric/repr_loss",
                            repr_loss.item(),
                            global_step=i
                        )

            # Print current progress to progress bar
            fmt = "{:.6f}".format
            loader.set_postfix(
                last=fmt(tracked_last),
                loss=fmt(tracked_loss),
                bad=fmt(evals[-1].detach().cpu().item() * 1000)
                if not args.baseline
                else 0
            )

            
            if args.export_dir:
                # export last inner optim of each input as csv (one input per row)
                if args.export_csv:
                    # the second to last element are the last of the inner optim 
                    for batch_i, p in enumerate(progress[-2]):
                        img_id = i * args.batch_size + batch_i
                        fname = loader.iterable.dataset.get_fname(img_id)
                        im_x, im_y = loader.iterable.dataset.get_imsize(img_id)

                        m = masks[-2][batch_i].cpu().detach().numpy().astype(bool)
                        p = p.cpu().detach().numpy()
                        p = p[:, m] * [[im_x], [im_y]]                        

                        sample_preds = [p[k%2, k//2] for k in range(p.shape[1] * 2)]
                        # remove values according to mask and add zeros to the end
                        # in stead
                        sample_preds += [0] * (len(m) * 2 - len(sample_preds))
                        predictions[fname] = sample_preds


                        # input_img = input[batch_i].detach().cpu()
                        # plt.scatter(p[0, :]*128, p[1, :]*128)
                        # plt.imshow(np.transpose(input_img, (1, 2, 0)))
                        # plt.show()

                        # if len(fname) == 8:
                        #     ds = "faces"
                        # else:
                        #     ds = "cats"
                        # img = mpimg.imread(ds + '/images/val/' + fname)
                        # plt.scatter(p[0, :]*img.shape[1], p[1, :]*img.shape[0])
                        # plt.imshow(img)
                        # plt.show()
                # Store predictions to be exported
                else:
                    if len(true_export) < args.export_n:
                        for p, m in zip(target_set, target_mask):
                            true_export.append(p.detach().cpu())
                        progress_steps = []
                        for pro, ms in zip(progress, masks):
                            # pro and ms are one step of the inner optim
                            # score boxes contains the list of predicted elements
                            # for one step
                            score_boxes = []
                            for p, m in zip(pro.cpu().detach(), ms.cpu().detach()):
                                score_box = torch.cat([m.unsqueeze(0), p], dim=0)
                                score_boxes.append(score_box)
                            progress_steps.append(score_boxes)
                        for b in zip(*progress_steps):
                            pred_export.append(b)

            # Plot predictions in Tensorboard
            if args.show and epoch % args.show_skip == 0 and not train:
                name = f"set/epoch-{epoch}/img-{i}"
                # thresholded set
                progress.append(progress[-1])
                masks.append((masks[-1] > 0.5).float())
                # target set
                if args.mask_feature:
                    # target set is augmented with masks, so remove them
                    progress.append(target_set[:, :-1])
                else:
                    progress.append(target_set)
                masks.append(target_mask)
                # intermediate sets

                for j, (s, ms) in enumerate(zip(progress, masks)):
                    if args.dataset == "clevr-state":
                        continue

                    if args.dataset.startswith("clevr"):
                        threshold = 0.5
                    else:
                        threshold = None

                    s, ms = utils.scatter_masked(
                        s,
                        ms,
                        binned=args.dataset.startswith("clevr"),
                        threshold=threshold
                    )

                    if j != len(progress) - 1:
                        tag_name = f"{name}"
                    else:
                        tag_name = f"{name}-target"

                    if args.dataset == "clevr-box":
                        img = input[0].detach().cpu()

                        writer.add_image_with_boxes(
                            tag_name,
                            img,
                            s.transpose(0, 1),
                            global_step=j
                        )
                    elif args.dataset == "cats" \
                            or args.dataset == "faces" \
                            or args.dataset == "wflw" \
                            or args.dataset == "merged":

                        img = input[0].detach().cpu()

                        fig = plt.figure()
                        plt.scatter(s[0, :]*128, s[1, :]*128)

                        plt.imshow(np.transpose(img, (1, 2, 0)))

                        writer.add_figure(tag_name, fig, global_step=j)
                    else:  # mnist
                        fig = plt.figure()
                        y, x = s
                        y = 1 - y
                        ms = ms.numpy()
                        rgba_colors = np.zeros((ms.size, 4))
                        rgba_colors[:, 2] = 1.0
                        rgba_colors[:, 3] = ms
                        plt.scatter(x, y, color=rgba_colors)
                        plt.axes().set_aspect("equal")
                        plt.xlim(0, 1)
                        plt.ylim(0, 1)
                        writer.add_figure(tag_name, fig, global_step=j)

        # Export predictions
        if args.export_dir and not args.export_csv:
            os.makedirs(f"{args.export_dir}/groundtruths", exist_ok=True)
            os.makedirs(f"{args.export_dir}/detections", exist_ok=True)
            for i, (gt, dets) in enumerate(zip(true_export, pred_export)):
                export_groundtruths_path = os.path.join(
                    args.export_dir,
                    "groundtruths",
                    f"{i}.txt"
                )

                with open(export_groundtruths_path, "w") as fd:
                    for box in gt.transpose(0, 1):
                        if (box == 0).all():
                            continue
                        s = "box " + " ".join(map(str, box.tolist()))
                        fd.write(s + "\n")

                if args.export_progress:
                    for step, det in enumerate(dets):
                        export_progress_path = os.path.join(
                            args.export_dir,
                            "detections",
                            f"{i}-step{step}.txt"
                        )

                        with open(export_progress_path, "w") as fd:
                            for sbox in det.transpose(0, 1):
                                s = f"box " + " ".join(map(str, sbox.tolist()))
                                fd.write(s + "\n")

                export_path = os.path.join(
                            args.export_dir,
                            "detections",
                            f"{i}.txt"
                        )
                with open(export_path, "w") as fd:
                    for sbox in dets[-1].transpose(0, 1):
                        s = f"box " + " ".join(map(str, sbox.tolist()))
                        fd.write(s + "\n")

    import subprocess

    git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"])
    # git_hash = "483igtrfiuey46"

    torch.backends.cudnn.benchmark = True

    metrics = {}

    start = time.time()

    if args.eval_only:
        tracker.new_epoch()
        with mp.Pool(10) as pool:
            run(
                net,
                test_loader,
                optimizer,
                train=False,
                epoch=0,
                pool=pool
            )

        metrics["test_loss"] = np.mean(tracker.data["test_loss"][-1])
        metrics["test_set_loss"] = np.mean(tracker.data["test_last"][-1])
    else:
        best_test_loss = float("inf")
        best_train_loss = float("inf")
        best_epoch = -1

        best_train_set_loss = float("inf")
        best_test_set_loss = float("inf")

        for epoch in range(args.epochs):
            tracker.new_epoch()
            with mp.Pool(10) as pool:
                run(
                    net,
                    train_loader,
                    optimizer,
                    train=True,
                    epoch=epoch,
                    pool=pool
                )
                if not args.train_only:
                    run(
                        net,
                        test_loader,
                        optimizer,
                        train=False,
                        epoch=epoch,
                        pool=pool
                    )

            epoch_test_loss = np.mean(tracker.data["test_loss"][-1])

            if epoch_test_loss < best_test_loss:
                print("new best loss")
                best_test_loss = epoch_test_loss
                # only save if the epoch has lower loss
                metrics["test_loss"] = epoch_test_loss
                metrics["train_loss"] = np.mean(tracker.data["train_loss"][-1])

                metrics["train_set_loss"] = np.mean(tracker.data["train_last"][-1])
                metrics["test_set_loss"] = np.mean(tracker.data["test_last"][-1])

                metrics["best_epoch"] = epoch

                results = {
                    "name": name + "_best",
                    "tracker": tracker.data,
                    "weights": net.state_dict()
                    if not args.multi_gpu
                    else net.module.state_dict(),
                    "args": vars(args),
                    "hash": git_hash,
                }

                torch.save(results, os.path.join("logs", name + "_best"))

        results = {
            "name": name + "_final",
            "tracker": tracker.data,
            "weights": net.state_dict()
            if not args.multi_gpu
            else net.module.state_dict(),
            "args": vars(args),
            "hash": git_hash,
        }
        torch.save(results, os.path.join("logs", name + "_final"))

    if args.export_csv and args.export_dir:

        cols = []
        # get number of rows from any value, which is what this loop iterates over
        for i in range(len(next(iter(predictions.values()))) // 2):
            for l in ['x', 'y']:
                cols.append(l + str(i))

        pd.DataFrame.from_dict(predictions, orient="index", columns=cols).to_csv(
            os.path.join(args.export_dir, 'predictions.csv'), 
            sep=',', 
            index_label="name"
        )

    took = time.time() - start
    print(f"Process took {took:.1f}s, avg {took/args.epochs:.1f} s/epoch.")

    # save hyper parameters to tensorboard for reference
    hparams = {k: v for k, v in vars(args).items() if v is not None}

    print(metrics)
    metrics = {
        "total_time": took,
        "avg_time_per_epoch": took/args.epochs
    }

    print("writing hparams")
    train_writer.add_hparams(hparams, metric_dict=metrics, name="hparams")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Process interrupted by user, emptying cache...")
        torch.cuda.empty_cache()
