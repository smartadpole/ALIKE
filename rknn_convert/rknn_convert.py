import yaml
from rknn.api import RKNN
import cv2
import os


CURRENT_DIR = os.path.dirname(__file__)
print(CURRENT_DIR)

_model_load_dict = {
    'caffe': 'load_caffe',
    'tensorflow': 'load_tensorflow',
    'tflite': 'load_tflite',
    'onnx': 'load_onnx',
    'darknet': 'load_darknet',
    'pytorch': 'load_pytorch',
    'mxnet': 'load_mxnet',
    'rknn': 'load_rknn',
    }

yaml_file = './config.yaml'


def main():
    with open(yaml_file, 'r') as F:
        config = yaml.load(F,Loader=yaml.FullLoader)
    # print('config is:')
    # print(config)

    model_type = config['running']['model_type']
    print('model_type is {}'.format(model_type))

    rknn = RKNN(verbose=True)

    print('--> config model')
    rknn.config(**config['config'])
    print('done')


    print('--> Loading model')
    load_function = getattr(rknn, _model_load_dict[model_type])
    ret = load_function(**config['parameters'][model_type])
    if ret != 0:
        print('Load mobilenet_v2 failed! Ret = {}'.format(ret))
        exit(ret)
    print('done')

    ####
    # print('hybrid_quantization')
    # ret = rknn.hybrid_quantization_step1(dataset=config['build']['dataset'])


    if model_type != 'rknn':
        print('--> Building model')
        ret = rknn.build(**config['build'])
        if ret != 0:
            print('Build mobilenet_v2 failed!')
            exit(ret)
    else:
        print('--> skip Building model step, cause the model is already rknn')


    if config['running']['export'] is True:
        print('--> Export RKNN model')
        ret = rknn.export_rknn(**config['export_rknn'])
        if ret != 0:
            print('Init runtime environment failed')
            exit(ret)
    else:
        print('--> skip Export model')


    if (config['running']['inference'] is True) or (config['running']['eval_perf'] is True):
        print('--> Init runtime environment')
        ret = rknn.init_runtime(**config['init_runtime'])
        if ret != 0:
            print('Init runtime environment failed')
            exit(ret)

        print('--> load img')
        img = cv2.imread(config['img']['path'])
        img = cv2.resize(img, (640, 384))
        print('img shape is {}'.format(img.shape))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        inputs = [img]


        if config['running']['inference'] is True:
            print('--> Running model')
            config['inference']['inputs'] = inputs
            #print(config['inference'])
            outputs = rknn.inference(inputs)
            #outputs = rknn.inference(config['inference'])
            print('len of output {}'.format(len(outputs)))
            [print('output shape is {}'.format(output.shape)) for output in outputs]
            print(outputs[0][0][0:2])
        else:
            print('--> skip inference')


        if config['running']['eval_perf'] is True:
            print('--> Begin evaluate model performance')
            config['inference']['inputs'] = inputs
            perf_results = rknn.eval_perf(inputs=[img])
        else:
            print('--> skip eval_perf')
    else:
        print('--> skip inference')
        print('--> skip eval_perf')

if __name__ == '__main__':
    main()
