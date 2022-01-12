import numpy as np
from scipy.ndimage import distance_transform_edt as distance
from skimage.segmentation import find_boundaries

def mask2rle(img, width, height):
    rle = []
    lastColor = 0;
    currentPixel = 0;
    runStart = -1;
    runLength = 0;

    for x in range(width):
        for y in range(height):
            currentColor = img[x][y]
            if currentColor != lastColor:
                if currentColor == 255:
                    runStart = currentPixel;
                    runLength = 1;
                else:
                    rle.append(str(runStart));
                    rle.append(str(runLength));
                    runStart = -1;
                    runLength = 0;
                    currentPixel = 0;
            elif runStart > -1:
                runLength += 1
            lastColor = currentColor;
            currentPixel += 1;

    return " ".join(rle)


def rle2mask(rle, width, height):
    mask = np.zeros(width * height)
    if rle == ' -1' or rle == '-1':
        return mask.reshape(width, height)

    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        current_position += start
        mask[current_position:current_position + lengths[index]] = 255
        current_position += lengths[index]

    return mask.reshape(width, height)


def compute_sdm_and_wmap(mask, is_norm):
    sdm = np.zeros(mask.shape)
    contour_map = np.ones(mask.shape)

    for i, m in enumerate(mask):
        posmask = np.array(m, dtype=np.bool)
        if posmask.any():  # 只有存在病灶的阳性样本才计算sdm
            negmask = ~posmask
            posdis = distance(posmask)
            negdis = distance(negmask)
            boundary = find_boundaries(posmask, mode='inner').astype(np.uint8)

            contour_map[i] += boundary

            if is_norm:
                sdf = (negdis - np.min(negdis)) / (np.max(negdis) - np.min(negdis)) - (posdis - np.min(posdis)) / (
                        np.max(posdis) - np.min(posdis))
            else:
                sdf = negdis - posdis
            sdf[boundary == 1] = 0

            sdm[i] = sdf

    w_map = contour_map - sdm

    return sdm, w_map
