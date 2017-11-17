#include <string>
#include <fstream>
#include <iostream>
#include <map>
#include <tuple>

#include <NvInfer.h>
#include <NvCaffeParser.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>

#include "common.hpp"

// reference: TensorRT-2.1.2/samples/sampleMNIST/sampleMNISTAPI.cpp

// intput width, input height, output size, input blob name, output blob name, weight file, mean file
typedef std::tuple<int, int, int, std::string, std::string, std::string, std::string> DATA_INFO;

// Our weight files are in a very simple space delimited format.
// [type] [size] <data x size in hex> 
static std::map<std::string, nvinfer1::Weights> loadWeights(const std::string& file)
{
    std::map<std::string, nvinfer1::Weights> weightMap;
	std::ifstream input(file);
	if (!input.is_open()) {
		fprintf(stderr, "Unable to load weight file: %s\n", file.c_str());
		return  weightMap;
	}

    int32_t count;
    input >> count;
	if (count <= 0) {
		fprintf(stderr, "Invalid weight map file: %d\n", count);
		return weightMap;
	}

    while(count--) {
    	nvinfer1:: Weights wt{nvinfer1::DataType::kFLOAT, nullptr, 0};
    	uint32_t type, size;
        std::string name;
        input >> name >> std::dec >> type >> size;
        wt.type = static_cast<nvinfer1::DataType>(type);
        if (wt.type == nvinfer1::DataType::kFLOAT) {
            uint32_t *val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
            for (uint32_t x = 0, y = size; x < y; ++x) {
                input >> std::hex >> val[x];
            }
            wt.values = val;
        } else if (wt.type == nvinfer1::DataType::kHALF) {
            uint16_t *val = reinterpret_cast<uint16_t*>(malloc(sizeof(val) * size));
            for (uint32_t x = 0, y = size; x < y; ++x) {
                input >> std::hex >> val[x];
            }
            wt.values = val;
        }
        wt.count = size;
        weightMap[name] = wt;
    }

    return weightMap;
}

// Creat the Engine using only the API and not any parser.
static nvinfer1::ICudaEngine* createMNISTEngine(unsigned int maxBatchSize, nvinfer1::IBuilder* builder, nvinfer1::DataType dt, const DATA_INFO& info)
{
	nvinfer1::INetworkDefinition* network = builder->createNetwork();

	//  Create input of shape { 1, 1, 28, 28 } with name referenced by INPUT_BLOB_NAME
	auto data = network->addInput(std::get<3>(info).c_str(), dt, nvinfer1::DimsCHW{ 1, std::get<1>(info), std::get<0>(info)});
	assert(data != nullptr);

	// Create a scale layer with default power/shift and specified scale parameter.
	float scale_param = 0.0125f;
	nvinfer1::Weights power{nvinfer1::DataType::kFLOAT, nullptr, 0};
	nvinfer1::Weights shift{nvinfer1::DataType::kFLOAT, nullptr, 0};
	nvinfer1::Weights scale{nvinfer1::DataType::kFLOAT, &scale_param, 1};
	auto scale_1 = network->addScale(*data,	nvinfer1::ScaleMode::kUNIFORM, shift, scale, power);
	assert(scale_1 != nullptr);

	// Add a convolution layer with 20 outputs and a 5x5 filter.
    std::map<std::string, nvinfer1::Weights> weightMap = loadWeights(std::get<5>(info));
	auto conv1 = network->addConvolution(*scale_1->getOutput(0), 20, nvinfer1::DimsHW{5, 5}, weightMap["conv1filter"], weightMap["conv1bias"]);
	assert(conv1 != nullptr);
	conv1->setStride(nvinfer1::DimsHW{1, 1});

	// Add a max pooling layer with stride of 2x2 and kernel size of 2x2.
	auto pool1 = network->addPooling(*conv1->getOutput(0), nvinfer1::PoolingType::kMAX, nvinfer1::DimsHW{2, 2});
	assert(pool1 != nullptr);
	pool1->setStride(nvinfer1::DimsHW{2, 2});

	// Add a second convolution layer with 50 outputs and a 5x5 filter.
	auto conv2 = network->addConvolution(*pool1->getOutput(0), 50, nvinfer1::DimsHW{5, 5}, weightMap["conv2filter"], weightMap["conv2bias"]);
	assert(conv2 != nullptr);
	conv2->setStride(nvinfer1::DimsHW{1, 1});

	// Add a second max pooling layer with stride of 2x2 and kernel size of 2x3>
	auto pool2 = network->addPooling(*conv2->getOutput(0), nvinfer1::PoolingType::kMAX, nvinfer1::DimsHW{2, 2});
	assert(pool2 != nullptr);
	pool2->setStride(nvinfer1::DimsHW{2, 2});

	// Add a fully connected layer with 500 outputs.
	auto ip1 = network->addFullyConnected(*pool2->getOutput(0), 500, weightMap["ip1filter"], weightMap["ip1bias"]);
	assert(ip1 != nullptr);

	// Add an activation layer using the ReLU algorithm.
	auto relu1 = network->addActivation(*ip1->getOutput(0), nvinfer1::ActivationType::kRELU);
	assert(relu1 != nullptr);

	// Add a second fully connected layer with 20 outputs.
	auto ip2 = network->addFullyConnected(*relu1->getOutput(0), std::get<2>(info), weightMap["ip2filter"], weightMap["ip2bias"]);
	assert(ip2 != nullptr);

	// Add a softmax layer to determine the probability.
	auto prob = network->addSoftMax(*ip2->getOutput(0));
	assert(prob != nullptr);
	prob->getOutput(0)->setName(std::get<4>(info).c_str());
	network->markOutput(*prob->getOutput(0));

	// Build the engine
	builder->setMaxBatchSize(maxBatchSize);
	builder->setMaxWorkspaceSize(1 << 20);

	auto engine = builder->buildCudaEngine(*network);
	// we don't need the network any more
	network->destroy();

	// Once we have built the cuda engine, we can release all of our held memory.
	for (auto &mem : weightMap) {
        free((void*)(mem.second.values));
    }

	return engine;
}

static int APIToModel(unsigned int maxBatchSize, // batch size - NB must be at least as large as the batch we want to run with)
		     nvinfer1::IHostMemory** modelStream, Logger logger, const DATA_INFO& info)
{
	// create the builder
	nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);

	// create the model to populate the network, then set the outputs and create an engine
	nvinfer1::ICudaEngine* engine = createMNISTEngine(maxBatchSize, builder, nvinfer1::DataType::kFLOAT, info);
	CHECK(engine != nullptr);

	// serialize the engine, then close everything down
	(*modelStream) = engine->serialize();
	engine->destroy();
	builder->destroy();

	return 0;
}

static int doInference(nvinfer1::IExecutionContext& context, float* input, float* output, int batchSize, const DATA_INFO& info)
{
	const nvinfer1::ICudaEngine& engine = context.getEngine();
	// input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
	// of these, but in this case we know that there is exactly one input and one output.
	CHECK(engine.getNbBindings() == 2);
	void* buffers[2];

	// In order to bind the buffers, we need to know the names of the input and output tensors.
	// note that indices are guaranteed to be less than IEngine::getNbBindings()
	int inputIndex = engine.getBindingIndex(std::get<3>(info).c_str()), 
	    outputIndex = engine.getBindingIndex(std::get<4>(info).c_str());

	// create GPU buffers and a stream
	checkCudaErrors(cudaMalloc(&buffers[inputIndex], batchSize * std::get<1>(info) * std::get<0>(info) * sizeof(float)));
	checkCudaErrors(cudaMalloc(&buffers[outputIndex], batchSize * std::get<2>(info) * sizeof(float)));

	cudaStream_t stream;
	checkCudaErrors(cudaStreamCreate(&stream));

	// DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
	checkCudaErrors(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * std::get<1>(info) * std::get<0>(info) * sizeof(float), cudaMemcpyHostToDevice, stream));
	context.enqueue(batchSize, buffers, stream, nullptr);
	checkCudaErrors(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * std::get<2>(info) * sizeof(float), cudaMemcpyDeviceToHost, stream));
	cudaStreamSynchronize(stream);

	// release the stream and the buffers
	cudaStreamDestroy(stream);
	checkCudaErrors(cudaFree(buffers[inputIndex]));
	checkCudaErrors(cudaFree(buffers[outputIndex]));

	return 0;
}

int test_mnist_api()
{
	Logger logger; // multiple instances of IRuntime and/or IBuilder must all use the same logger
	// stuff we know about the network and the caffe input/output blobs
	const DATA_INFO info(28, 28, 10, "data", "prob", "models/mnistapi.wts", "models/mnist_mean.binaryproto");

	// create a model using the API directly and serialize it to a stream
    nvinfer1::IHostMemory* modelStream{ nullptr };
    APIToModel(1, &modelStream, logger, info);

	// parse the mean file produced by caffe and subtract it from the image
	nvcaffeparser1::ICaffeParser* parser = nvcaffeparser1::createCaffeParser();
	nvcaffeparser1::IBinaryProtoBlob* meanBlob = parser->parseBinaryProto(std::get<6>(info).c_str());
	parser->destroy();
	const float* meanData = reinterpret_cast<const float*>(meanBlob->getData());

	nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);
	nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(modelStream->data(), modelStream->size(), nullptr);
	nvinfer1::IExecutionContext* context = engine->createExecutionContext();

	uint8_t fileData[std::get<1>(info) * std::get<0>(info)];
	const std::string image_path{ "images/digit/" };
	for (int i = 0; i < 10; ++i) {
		const std::string image_name = image_path + std::to_string(i) + ".png";
		cv::Mat mat = cv::imread(image_name, 0);
		if (!mat.data) {
			fprintf(stderr, "read image fail: %s\n", image_name.c_str());
			return -1;
		}

		cv::resize(mat, mat, cv::Size(std::get<0>(info), std::get<1>(info)));
		mat.convertTo(mat, CV_32FC1);

		float data[std::get<1>(info)*std::get<0>(info)];
		const float* p = (float*)mat.data;
		for (int j = 0; j < std::get<1>(info)*std::get<0>(info); ++j) {
			data[j] = p[j] - meanData[j];
		}

		// run inference
		float prob[std::get<2>(info)];
		doInference(*context, data, prob, 1, info);

		float val{-1.f};
		int idx{-1};

		for (int t = 0; t < std::get<2>(info); ++t) {
			if (val < prob[t]) {
				val = prob[t];
				idx = t;
			}
		}

		fprintf(stdout, "expected value: %d, actual value: %d, probability: %f\n", i, idx, val);
	}

	meanBlob->destroy();
    if (modelStream) modelStream->destroy();
	// destroy the engine
	context->destroy();
	engine->destroy();
	runtime->destroy();

	return 0;
}
