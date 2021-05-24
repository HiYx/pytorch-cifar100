import timeit
import numpy as np
import cv2
import tensorrt as trt
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
datatype = trt.float32

PIXEL_MEANS = np.array([[[104., 117., 123.]]], dtype=np.float32)
DEPLOY_ENGINE = 'trtModel.cache'
ENGINE_SHAPE0 = (3, 32, 32)
ENGINE_SHAPE1 = (1, 512)
RESIZED_SHAPE = (32, 32)

def classify(img, net, labels):
    """Classify 1 image (crop)."""
    crop = img

    # preprocess the image crop
    crop = cv2.resize(crop, RESIZED_SHAPE)
    crop = crop.astype(np.float32) - PIXEL_MEANS
    crop = crop.transpose((2, 0, 1))  # HWC -> CHW

    # inference the (cropped) image
    tic = timeit.default_timer()
    out = net.forward(crop[None])  # add 1 dimension to 'crop' as batch
    toc = timeit.default_timer()
    print('{:.3f}s'.format(toc-tic))

    # output top 3 predicted scores and class labels
    # out_prob = np.squeeze(out['prob'][0])
    # top_inds = out_prob.argsort()[::-1][:3]
    # return (out_prob[top_inds], labels[top_inds])


def loop_and_classify(cam, net, labels):
    """Continuously capture images from camera and do classification."""
    img = cam
    if img is None:
        break
    classify(img, net, labels)

def main():
    args = parse_args()
    labels = np.loadtxt('googlenet/synset_words.txt', str, delimiter='\t')
    # initialize the tensorrt googlenet engine
    net = PyTrtGooglenet(DEPLOY_ENGINE, ENGINE_SHAPE0, ENGINE_SHAPE1)
    
    
    loop_and_classify(cam, net, labels)



if __name__ == '__main__':
    main()
