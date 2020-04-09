# yolov3-tiny-onnx-TensorRT
## Requirements

- TensorRT 6.0
- VS 2015/Clion
- Cuda 9.0 + Cudnn 7.6



## Model Converter

Convert your Darknet yolov3-tiny model to onnx，please follow these steps：

## Requirements

	python=2.7
	numpy=1.16.1
	onnx=1.4.1 (important)
	pycuda=2019.1.1
	Pillow=6.1.0
	wget=3.2

## custom settings

	data_processing.py:
		line14: LABEL_FILE_PATH = '/home/nvidia/yolov3-tiny2onnx2trt/coco_labels.txt'
		line19: CATEGORY_NUM = 80
	
	yolov3_to_onnx.py:
		line778: img_size = 416
		line784: cfg_file_path = '/home/nvidia/yolov3-tiny2onnx2trt/yolov3-tiny.cfg'
		line811: weights_file_path = '/home/nvidia/yolov3-tiny2onnx2trt/yolov3-tiny.weights'
		line826: output_file_path = 'yolov3-tiny.onnx'
	
	onnx_to_tensorrt.py:
		line39: input_size = 416
		line40: batch_size = 1
		line42~line46:
		    onnx_file_path = 'yolov3-tiny.onnx'
		    engine_file_path = 'yolov3-tiny.trt'
		    input_file_list = '/home/nvidia/yolov3-tiny2onnx2trt/imagelist.txt'
		    IMAGE_PATH = '/home/nvidia/yolov3-tiny2onnx2trt/images/'
		    save_path = '/home/nvidia/yolov3-tiny2onnx2trt/'
## notes (very important!)

	0.The onnx version must be 1.4.1. If it is not, please run the commands:
		pip uninstall onnx
		pip install onnx==1.4.1
	
	1.The cfg-file's last line must be a blank line. You should press Enter to add a blank line if there is no blank line at the end of the file.

## steps

	0.Put your .weights file in the folder
		|-yolov3-tiny2onnx2trt
			|-yolov3-tiny.weights
	
	1.Change your settings as "#custom settings"
	
	2.Run commands:
		cd yolov3-tiny2onnx2trt
		python yolov3_to_onnx.py
	
		you will get a yolov3-tiny.onnx file
	
	3.Run commands:	
	  	python onnx_to_tensorrt.py:
	
		you will get a yolov3-tiny.trt file and some inferenced images.


# TensorRT FP32 Inference

- run yolov3-tiny-trt-fp32.cpp（ You can modify the number of categories by yourself ）。

- The visualization results are as follows：



![在这里插入图片描述](https://img-blog.csdnimg.cn/20200409142919546.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70#pic_center)





![在这里插入图片描述](https://img-blog.csdnimg.cn/20200409143133305.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70#pic_center)



![在这里插入图片描述](https://img-blog.csdnimg.cn/20200409143229456.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70#pic_center)



# TensorRT INT8 Calibaration

- Prepare calibaration data(*.txt)，like this：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200409151326680.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

- Create a class that inherits INT8EntropyCalibrator, the code is as follows：



```c++
namespace nvinfer1 {
	class int8EntroyCalibrator : public nvinfer1::IInt8EntropyCalibrator {
	public:
		int8EntroyCalibrator(const int &bacthSize,
			const std::string &imgPath,
			const std::string &calibTablePath);

		virtual ~int8EntroyCalibrator();

		int getBatchSize() const override { return batchSize; }

		bool getBatch(void *bindings[], const char *names[], int nbBindings) override;

		const void *readCalibrationCache(std::size_t &length) override;

		void writeCalibrationCache(const void *ptr, std::size_t length) override;

	private:

		bool forwardFace;

		int batchSize;
		size_t inputCount;
		size_t imageIndex;

		std::string calibTablePath;
		std::vector<std::string> imgPaths;

		float *batchData{ nullptr };
		void  *deviceInput{ nullptr };



		bool readCache;
		std::vector<char> calibrationCache;
	};

	int8EntroyCalibrator::int8EntroyCalibrator(const int &bacthSize, const std::string &imgPath,
		const std::string &calibTablePath) :batchSize(bacthSize), calibTablePath(calibTablePath), imageIndex(0), forwardFace(
			false) {
		int inputChannel = 3;
		int inputH = 416;
		int inputW = 416;
		inputCount = bacthSize*inputChannel*inputH*inputW;
		std::fstream f(imgPath);
		if (f.is_open()) {
			std::string temp;
			while (std::getline(f, temp)) imgPaths.push_back(temp);
		}
		int len = imgPaths.size();
		for (int i = 0; i < len; i++) {
			cout << imgPaths[i] << endl;
		}
		batchData = new float[inputCount];
		CHECK(cudaMalloc(&deviceInput, inputCount * sizeof(float)));
	}

	int8EntroyCalibrator::~int8EntroyCalibrator() {
		CHECK(cudaFree(deviceInput));
		if (batchData)
			delete[] batchData;
	}

	bool int8EntroyCalibrator::getBatch(void **bindings, const char **names, int nbBindings) {
		cout << imageIndex << " " << batchSize << endl;
		cout << imgPaths.size() << endl;
		if (imageIndex + batchSize > int(imgPaths.size()))
			return false;
		// load batch
		float* ptr = batchData;
		for (size_t j = imageIndex; j < imageIndex + batchSize; ++j)
		{
			//cout << imgPaths[j] << endl;
			Mat img = cv::imread(imgPaths[j]);
			vector<float>inputData = prepareImage(img);
			cout << inputData.size() << endl;
			cout << inputCount << endl;
			if ((int)(inputData.size()) != inputCount)
			{
				std::cout << "InputSize error. check include/ctdetConfig.h" << std::endl;
				return false;
			}
			assert(inputData.size() == inputCount);
			int len = (int)(inputData.size());
			memcpy(ptr, inputData.data(), len * sizeof(float));

			ptr += inputData.size();
			std::cout << "load image " << imgPaths[j] << "  " << (j + 1)*100. / imgPaths.size() << "%" << std::endl;
		}
		imageIndex += batchSize;
		CHECK(cudaMemcpy(deviceInput, batchData, inputCount * sizeof(float), cudaMemcpyHostToDevice));
		bindings[0] = deviceInput;
		return true;
	}
	const void* int8EntroyCalibrator::readCalibrationCache(std::size_t &length)
	{
		calibrationCache.clear();
		std::ifstream input(calibTablePath, std::ios::binary);
		input >> std::noskipws;
		if (readCache && input.good())
			std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(),
				std::back_inserter(calibrationCache));

		length = calibrationCache.size();
		return length ? &calibrationCache[0] : nullptr;
	}

	void int8EntroyCalibrator::writeCalibrationCache(const void *cache, std::size_t length)
	{
		std::ofstream output(calibTablePath, std::ios::binary);
		output.write(reinterpret_cast<const char*>(cache), length);
	}
}
```



- Change onnxToTRTModel function in yolov3-tiny-trt-fp32.cpp，the code is as follows：

```c++
bool onnxToTRTModel(const std::string& modelFile,
	const std::string& filename,  
	IHostMemory*& trtModelStream) // output buffer for the TensorRT model
{
	IBuilder* builder = createInferBuilder(gLogger.getTRTLogger());
	assert(builder != nullptr);
	nvinfer1::INetworkDefinition* network = builder->createNetwork();

	if (!builder->platformHasFastInt8()) return false;

	auto parser = nvonnxparser::createParser(*network, gLogger.getTRTLogger());


	//config->setPrintLayerInfo(true);
	//parser->reportParsingInfo();

	if (!parser->parseFromFile(modelFile.c_str(), static_cast<int>(gLogger.getReportableSeverity())))
	{
		gLogError << "Failure while parsing ONNX file" << std::endl;
		return false;
	}

	
	builder->setMaxBatchSize(BATCH_SIZE);
	builder->setMaxWorkspaceSize(1 << 30);

	nvinfer1::int8EntroyCalibrator *calibrator = nullptr;
	if (calibFile.size()>0) calibrator = new nvinfer1::int8EntroyCalibrator(BATCH_SIZE, calibFile, "F:/TensorRT-6.0.1.5/data/v3tiny/calib.table");


	//builder->setFp16Mode(true);
	std::cout << "setInt8Mode" << std::endl;
	if (!builder->platformHasFastInt8())
		std::cout << "Notice: the platform do not has fast for int8" << std::endl;
	builder->setInt8Mode(true);
	builder->setInt8Calibrator(calibrator);
	/*if (gArgs.runInInt8)
	{
		samplesCommon::setAllTensorScales(network, 127.0f, 127.0f);
	}*/
	//samplesCommon::setAllTensorScales(network, 1.0f, 1.0f);
	cout << "start building engine" << endl;
	ICudaEngine* engine = builder->buildCudaEngine(*network);
	cout << "build engine done" << endl;
	assert(engine);
	if (calibrator) {
		delete calibrator;
		calibrator = nullptr;
	}
	parser->destroy();

	trtModelStream = engine->serialize();

	nvinfer1::IHostMemory* data = engine->serialize();
	std::ofstream file;
	file.open(filename, std::ios::binary | std::ios::out);
	cout << "writing engine file..." << endl;
	file.write((const char*)data->data(), data->size());
	cout << "save engine file done" << endl;
	file.close();

	engine->destroy();
	network->destroy();
	builder->destroy();

	return true
```



- Finally you can get a INT8 TensorRT model，enjoy it。



# Accuracy And Speed

- GTX 1050 Ti

| YOLOV3-Tiny TRT模型 | mAP(50) | Inference Time |
| ------------------- | ------- | -------------- |
| FP32                | 95.0%   | 42ms           |
| INT8                | 95.0%   | 10ms           |



# Reference

- https://github.com/zombie0117/yolov3-tiny-onnx-TensorRT
- https://mp.weixin.qq.com/s/rYuodkH-tf-q4uZ0QAkuAw
- https://mp.weixin.qq.com/s/huP2J565irXXU7SSIk-Hwg
- https://mp.weixin.qq.com/s/9WKJi4AnOFKKqvK8R9ph1g
- https://mp.weixin.qq.com/s/QcotYLHVVkf5sEvgKZKemg
- https://mp.weixin.qq.com/s/WiVhlR9-rpe-O9J9ULc_bA