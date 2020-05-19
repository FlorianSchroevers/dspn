import argparse
import itertools
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import data
from PIL import Image


parser = argparse.ArgumentParser()
parser.add_argument("n", type=int, nargs="*")
parser.add_argument("--keep", type=int, nargs="*")
args = parser.parse_args()

matplotlib.rc("text", usetex=True)
params = {
    "text.latex.preamble": [r"\usepackage{bm,amsmath,mathtools,amssymb}"]
}
plt.rcParams.update(params)

base_path = "clevr/images/val"
val_images = sorted(os.listdir(base_path))


def take(iterable, n):
    li = []
    for _ in range(n):
        li.append(next(iterable))
    return li


def load_file(path):
    with open(path) as fd:
        for f in fd:
            tokens = iter(f.strip().split(" "))
            take(tokens, 1)
            if "detect" in path:
                score = float(take(tokens, 1)[0])
                if score < 0.5:
                    continue
            else:
                score = 1.0
            coord = take(tokens, 3)
            material = np.argmax(take(tokens, 2))
            color = np.argmax(take(tokens, 8))
            shape = np.argmax(take(tokens, 3))
            size = np.argmax(take(tokens, 2))

            yield (
                "({:.2f}, {:.2f}, {:.2f})".format(
                    *map(lambda x: 3 * float(x), coord)
                ),
                data.CLASSES["size"][size],
                data.CLASSES["color"][color],
                data.CLASSES["material"][material],
                data.CLASSES["shape"][shape],
            )


indices_to_use = args.keep
indices_to_use.append(-2)
indices_to_use.append(-1)

plt.figure(figsize=(12, 4))
for j, index in enumerate(args.n):
    progress = []

    for i in range(31):
        points_path = os.path.join(
            "out",
            "clevr-state",
            "dspn-clevr-state-1-30",
            "detections",
            f"{index}-step{i}.txt"
        )
        points = list(load_file(points_path))
        progress.append(points)

    groundtruths_path = os.path.join(
        "out",
        "clevr-state",
        "base-clevr-state-1-10",
        "groundtruths",
        f"{index}.txt"
    )

    progress.append(list(load_file(groundtruths_path)))

    detections_path = os.path.join(
        "out",
        "clevr-state",
        "base-clevr-state-1-10",
        "detections",
        f"{index}.txt"
    )

    progress.append(list(load_file(detections_path)))

    img = Image.open(os.path.join(base_path, val_images[int(index)]))
    img = img.resize((128, 128), Image.LANCZOS)
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(f"img-{j}.pdf", bbox_inches="tight")

    matrix = []
    for i, progress_n in enumerate(indices_to_use):
        column = []
        step = progress[progress_n]

        if progress_n == -2:
            header = r"True $\bm{Y}$"
        elif progress_n == -1:
            header = r"Baseline"
        else:
            header = r"$\hat{\bm{Y}}^{(" + str(progress_n) + ")}$"
        column.append(header)
        for object in sorted(
            step, key=lambda x: [
                float(x.strip()) for x in x[0][1:-1].split(",")
            ]
        ):
            column.append(object[0])
            column.append(" ".join(object[1:]))
        matrix.append(column)
    # transpose
    matrix = itertools.zip_longest(*matrix, fillvalue="")

    # make an attribute red if it isn't correct
    matrix = [
        [
            " ".join(
                (
                    r"\textcolor{red}{" + attribute + "}"
                    if attribute != correct_attribute
                    else attribute
                )
                for attribute, correct_attribute in zip(
                    state.split(" "), row[-2].split(" ")
                )
            )
            for state in row
        ]
        if "small" in row[-2] or "large" in row[-2]
        else row
        for row in matrix
    ]

    matrix = [" & ".join(row) for row in matrix]
    # format into table
    template = r"""
\includegraphics[width=0.22\linewidth]{{img-{}}}
\begin{{tabular}}{}
\toprule
{}\\
\midrule
{}\\
\bottomrule
\end{{tabular}}
"""
    table = template.format(
        j, "{" + "c" * len(indices_to_use) + "}", matrix[0], "\\\\\n".join(
            matrix[1:]
        )
    )
    print(table)
