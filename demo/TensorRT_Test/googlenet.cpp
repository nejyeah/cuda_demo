#include <iostream>
#include <tuple>
#include <string>
#include <vector>
#include <algorithm>

#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <NvCaffeParser.h>

#include "common.hpp"

// reference: TensorRT-2.1.2/samples/sampleMNIST/sampleGoogleNet.cpp

namespace {
// batch size, timing iterations, input blob name, output blob name, deploy file, model file
typedef std::tuple<int, int, std::string, std::string, std::string , std::string> DATA_INFO;  

struct Profiler : public nvinfer1::IProfiler {
	typedef std::pair<std::string, float> Record;
	std::vector<Record> mProfile;
	int timing_iterations {1};

	void setTimeIterations(int iteration)
	{
		timing_iterations = iteration;
	}

	virtual void reportLayerTime(const char* layerName, float ms)
	{
		auto record = std::find_if(mProfile.begin(), mProfile.end(), [&](const Record& r){ return r.first == layerName; });
		if (record == mProfile.end())
			mProfile.push_back(std::make_pair(layerName, ms));
		else
			record->second += ms;
	}

	void printLayerTimes()
	{
		float totalTime = 0;
		for (size_t i = 0; i < mProfile.size(); ++i) {
			fprintf(stdout, "%s %4.3fms\n", mProfile[i].first.c_str(), mProfile[i].second / timing_iterations);
			totalTime += mProfile[i].second;
		}
		fprintf(stdout, "Time over all layers: %4.3f\n", totalTime / timing_iterations);
	}

};

int caffeToGIEModel(const std::string& deployFile,		// name for caffe prototxt
					 const std::string& modelFile,				// name for model 
					 const std::vector<std::string>& outputs,   // network outputs
					 unsigned int maxBatchSize,					// batch size - NB must be at least as large as the batch we want to run with)
					 nvinfer1::IHostMemory *&gieModelStream, Logger logger)
{
	// create API root class - must span the lifetime of the engine usage
	nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);
	nvinfer1::INetworkDefinition* network = builder->createNetwork();

	// parse the caffe model to populate the network, then set the outputs
	nvcaffeparser1::ICaffeParser* parser = nvcaffeparser1::createCaffeParser();

	bool useFp16 = builder->platformHasFastFp16();

	nvinfer1::DataType modelDataType = useFp16 ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT; // create a 16-bit model if it's natively supported
	const nvcaffeparser1::IBlobNameToTensor* blobNameToTensor = parser->parse(deployFile.c_str(), modelFile.c_str(), *network, modelDataType);
	CHECK(blobNameToTensor != nullptr);

	// the caffe file has no notion of outputs, so we need to manually say which tensors the engine should generate	
	for (auto& s : outputs)
		network->markOutput(*blobNameToTensor->find(s.c_str()));

	// Build the engine
	builder->setMaxBatchSize(maxBatchSize);
	builder->setMaxWorkspaceSize(16 << 20);

	// set up the network for paired-fp16 format if available
	if(useFp16)
		builder->setHalf2Mode(true);

	nvinfer1::ICudaEngine* engine = builder->buildCudaEngine(*network);
	CHECK(engine != nullptr);

	// we don't need the network any more, and we can destroy the parser
	network->destroy();
	parser->destroy();

	// serialize the engine, then close everything down
	gieModelStream = engine->serialize();
	engine->destroy();
	builder->destroy();
	nvcaffeparser1::shutdownProtobufLibrary();

	return 0;
}

int timeInference(nvinfer1::ICudaEngine* engine, const DATA_INFO& info, Profiler* profiler)
{
	// input and output buffer pointers that we pass to the engine - the engine requires exactly ICudaEngine::getNbBindings(),
	// of these, but in this case we know that there is exactly one input and one output.
	CHECK(engine->getNbBindings() == 2);
	void* buffers[2];

	// In order to bind the buffers, we need to know the names of the input and output tensors.
	// note that indices are guaranteed to be less than ICudaEngine::getNbBindings()
	int inputIndex = engine->getBindingIndex(std::get<2>(info).c_str()), outputIndex = engine->getBindingIndex(std::get<3>(info).c_str());

	// allocate GPU buffers
	nvinfer1::DimsCHW inputDims = static_cast<nvinfer1::DimsCHW&&>(engine->getBindingDimensions(inputIndex)), outputDims = static_cast<nvinfer1::DimsCHW&&>(engine->getBindingDimensions(outputIndex));
	size_t inputSize = std::get<0>(info) * inputDims.c() * inputDims.h() * inputDims.w() * sizeof(float);
	size_t outputSize = std::get<0>(info) * outputDims.c() * outputDims.h() * outputDims.w() * sizeof(float);

	cudaMalloc(&buffers[inputIndex], inputSize);
	cudaMalloc(&buffers[outputIndex], outputSize);

	nvinfer1::IExecutionContext* context = engine->createExecutionContext();
	context->setProfiler(profiler);

	// zero the input buffer
	cudaMemset(buffers[inputIndex], 0, inputSize);

	for (int i = 0; i < std::get<1>(info); ++i)
		context->execute(std::get<0>(info), buffers);

	// release the context and buffers
	context->destroy();
	cudaFree(buffers[inputIndex]);
	cudaFree(buffers[outputIndex]);

	return 0;
}

} // namespace

int test_googlenet()
{
	fprintf(stdout, "Building and running a GPU inference engine for GoogleNet, N=4...\n");

	// stuff we know about the network and the caffe input/output blobs
	DATA_INFO info(4, 1000, "data", "prob", "models/googlenet.prototxt", "models/googlenet.caffemodel");
	Logger logger;

	// parse the caffe model and the mean file
   	nvinfer1::IHostMemory* gieModelStream{ nullptr };
	caffeToGIEModel(std::get<4>(info), std::get<5>(info), std::vector<std::string>{std::get<3>(info)}, std::get<0>(info), gieModelStream, logger);

	// create an engine
	nvinfer1::IRuntime* infer = nvinfer1::createInferRuntime(logger);
	nvinfer1::ICudaEngine* engine = infer->deserializeCudaEngine(gieModelStream->data(), gieModelStream->size(), nullptr);

	fprintf(stdout, "Bindings after deserializing:\n"); 
	for (int bi = 0; bi < engine->getNbBindings(); bi++) { 
		if (engine->bindingIsInput(bi) == true) { 
			fprintf(stdout, "Binding %d (%s): Input.\n",  bi, engine->getBindingName(bi)); 
		} else { 
			fprintf(stdout, "Binding %d (%s): Output.\n", bi, engine->getBindingName(bi)); 
		} 
	} 

	Profiler profiler;
	profiler.setTimeIterations(std::get<1>(info));

	// run inference with null data to time network performance
	timeInference(engine,  info, &profiler);

	engine->destroy();
	infer->destroy();

	profiler.printLayerTimes();

	fprintf(stdout, "Done.\n");

	return 0;
}
