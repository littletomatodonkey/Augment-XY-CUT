import numpy as np
import random


def xy_cut(bboxes, direction="x"):
    result = []
    K = len(bboxes)
    indexes = range(K)
    if len(bboxes) <= 0:
        return result
    if direction == "x":
        # x first
        sorted_ids = sorted(indexes, key=lambda k: (bboxes[k][0], bboxes[k][1]))
        sorted_boxes = sorted(bboxes, key=lambda x: (x[0], x[1]))
        next_dir = "y"
    else:
        sorted_ids = sorted(indexes, key=lambda k: (bboxes[k][1], bboxes[k][0]))
        sorted_boxes = sorted(bboxes, key=lambda x: (x[1], x[0]))
        next_dir = "x"

    curr = 0
    np_bboxes = np.array(sorted_boxes)
    for idx in range(len(sorted_boxes)):
        if direction == "x":
            # a new seg path
            if idx != K - 1 and sorted_boxes[idx][2] < sorted_boxes[idx + 1][0]:
                rel_res = xy_cut(sorted_boxes[curr:idx + 1], next_dir)
                result += [sorted_ids[i + curr] for i in rel_res]
                curr = idx + 1
        else:
            # a new seg path
            if idx != K - 1 and sorted_boxes[idx][3] < sorted_boxes[idx + 1][1]:
                rel_res = xy_cut(sorted_boxes[curr:idx + 1], next_dir)
                result += [sorted_ids[i + curr] for i in rel_res]
                curr = idx + 1

    result += sorted_ids[curr:idx + 1]
    return result


def augment_xy_cut(bboxes,
                   direction="x",
                   lambda_x=0.5,
                   lambda_y=0.5,
                   theta=5,
                   aug=False):
    if aug is True:
        for idx in range(len(bboxes)):
            vx = np.random.normal(loc=0, scale=1)
            vy = np.random.normal(loc=0, scale=1)
            if np.abs(vx) >= lambda_x:
                bboxes[idx][0] += round(theta * vx)
                bboxes[idx][2] += round(theta * vx)
            if np.abs(vy) >= lambda_y:
                bboxes[idx][1] += round(theta * vy)
                bboxes[idx][3] += round(theta * vy)
            bboxes[idx] = [max(0, i) for i in bboxes[idx]]
    res_idx = xy_cut(bboxes, direction=direction)
    res_bboxes = [bboxes[idx] for idx in res_idx]
    return res_idx, res_bboxes


# debug
# box info for FUNSD/testing_data/images/87428306.png
bboxes = [[115, 207, 127, 215], [112, 349, 155, 358], [123, 485, 144, 494],
          [127, 162, 158, 173], [119, 885, 156, 893], [190, 909, 221, 918],
          [154, 909, 186, 918], [115, 912, 152, 921], [646, 643, 667, 731],
          [511, 880, 530, 891], [258, 119, 545, 136], [114, 152, 175, 162],
          [508, 147, 556, 158], [115, 179, 171, 188], [128, 184, 547, 202],
          [128, 219, 569, 237], [130, 234, 588, 250], [128, 249, 554, 265],
          [128, 262, 580, 277], [115, 287, 178, 298], [128, 305, 234, 317],
          [126, 361, 215, 370], [127, 369, 536, 386], [126, 385, 349, 399],
          [127, 436, 567, 454], [127, 452, 400, 466], [128, 504, 568, 520],
          [127, 519, 401, 535], [116, 574, 205, 584], [127, 598, 516, 615],
          [115, 635, 220, 644], [126, 680, 517, 698], [116, 739, 206, 751],
          [135, 765, 158, 774], [171, 763, 200, 774], [214, 761, 262, 773],
          [273, 761, 309, 773], [119, 790, 312, 802], [372, 890, 425, 903],
          [652, 904, 700, 915]]

res_idx, res_bboxes = augment_xy_cut(bboxes, direction="y")
print(res_idx)
res_idx, res_bboxes = augment_xy_cut(bboxes, direction="x")
print(res_idx)
