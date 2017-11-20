#include <assert.h>
#include <string>
#include <string.h>
#include <fstream>
#include <iostream>
#include <tuple>
#include <map>
#include <sstream>
#include <vector>
#include <algorithm>

#include <NvInfer.h>
#include <NvUtils.h>
#include <cuda_runtime_api.h>

#include "common.hpp"

// reference: TensorRT-2.1.2/samples/sampleMNIST/sampleCharRNN.cpp
// demonstrates how to generate a simple RNN based on the charRNN network using the PTB dataset

namespace {

// Information describing the network:
// int: layer count, batch size, hidden size, seq size, data size, output size
// string: input blob name, hidden in blob name, cell in blob name, output blob name, hidden out blob name, cell out blob name
typedef std::tuple<int, int, int, int, int, int, std::string, std::string, std::string, std::string, std::string, std::string> NET_INFO;

// These mappings came from training with tensorflow 0.12.1
static std::map<char, int> char_to_id{{'#', 40},
    { '$', 31}, { '\'', 28}, { '&', 35}, { '*', 49},
    { '-', 32}, { '/', 48}, { '.', 27}, { '1', 37},
    { '0', 36}, { '3', 39}, { '2', 41}, { '5', 43},
    { '4', 47}, { '7', 45}, { '6', 46}, { '9', 38},
    { '8', 42}, { '<', 22}, { '>', 23}, { '\0', 24},
    { 'N', 26}, { '\\', 44}, { ' ', 0}, { 'a', 3},
    { 'c', 13}, { 'b', 20}, { 'e', 1}, { 'd', 12},
    { 'g', 18}, { 'f', 15}, { 'i', 6}, { 'h', 9},
    { 'k', 17}, { 'j', 30}, { 'm', 14}, { 'l', 10},
    { 'o', 5}, { 'n', 4}, { 'q', 33}, { 'p', 16},
    { 's', 7}, { 'r', 8}, { 'u', 11}, { 't', 2},
    { 'w', 21}, { 'v', 25}, { 'y', 19}, { 'x', 29},
    { 'z', 34}
};

// A mapping from index to character.
static std::vector<char> id_to_char{{' ', 'e', 't', 'a',
    'n', 'o', 'i', 's', 'r', 'h', 'l', 'u', 'd', 'c',
    'm', 'f', 'p', 'k', 'g', 'y', 'b', 'w', '<', '>',
    '\0', 'v', 'N', '.', '\'', 'x', 'j', '$', '-', 'q',
    'z', '&', '0', '1', '9', '3', '#', '2', '8', '5',
    '\\', '7', '6', '4', '/', '*'}};

// Our weight files are in a very simple space delimited format.
std::map<std::string, nvinfer1::Weights> loadWeights(const std::string& file)
{
    std::map<std::string, nvinfer1::Weights> weightMap;
    std::ifstream input(file);
    if (!input.is_open()) { fprintf(stderr, "Unable to load weight file: %s\n", file.c_str()); return weightMap;}
    int32_t count;
    input >> count;
    if (count <= 0) { fprintf(stderr, "Invalid weight map file: %d\n", count); return weightMap; }
    while (count--) {
        nvinfer1::Weights wt{nvinfer1::DataType::kFLOAT, nullptr, 0};
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

// Reshape plugin to feed RNN into FC layer correctly.
class Reshape : public nvinfer1::IPlugin {
public:
	Reshape(size_t size) : mSize(size) {} 
	Reshape(const void*buf, size_t size)
    {
        assert(size == sizeof(mSize));
        mSize = *static_cast<const size_t*>(buf);
    }

	int getNbOutputs() const override {	return 1; }
	int initialize() override {	return 0; }
	void terminate() override {}
	size_t getWorkspaceSize(int) const override { return 0;	}

	int enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream)
    {
       cudaMemcpyAsync(static_cast<float*>(outputs[0]),
                   static_cast<const float*>(inputs[0]),
                   sizeof(float) * mSize * batchSize, cudaMemcpyDefault, stream);
        return 0;
    }

	size_t getSerializationSize() override
    {
        return sizeof(mSize);
    }

	void serialize(void* buffer) override
    {
        (*static_cast<size_t*>(buffer)) = mSize;

    }

	void configure(const nvinfer1::Dims*, int, const nvinfer1::Dims*, int, int)	override { }

    // The RNN outputs in {L, N, C}, but FC layer needs {C, 1, 1}, so we can convert RNN
    // output to {L*N, C, 1, 1} and TensorRT will handle the rest.
	nvinfer1::Dims getOutputDimensions(int index, const nvinfer1::Dims* inputs, int nbInputDims) override
	{
        assert(nbInputDims == 1 && index == 0 && inputs[index].nbDims == 3);
		return nvinfer1::DimsNCHW(inputs[index].d[1] * inputs[index].d[0], inputs[index].d[2], 1, 1);
	}

private:
    size_t mSize{0};
};

class PluginFactory : public nvinfer1::IPluginFactory
{
public:
	// deserialization plugin implementation
	nvinfer1::IPlugin* createPlugin(const char* layerName, const void* serialData, size_t serialLength) override
	{
        assert(!strncmp(layerName, "reshape", 7));
        if (!mPlugin) mPlugin = new Reshape(serialData, serialLength);
        return mPlugin;
    }

    void destroyPlugin()
    {
        if (mPlugin) delete mPlugin;
        mPlugin = nullptr;
    }

private:
    Reshape *mPlugin{nullptr};
}; // PluginFactory
	
// TensorFlow weight parameters for BasicLSTMCell
nvinfer1::Weights convertRNNWeights(nvinfer1::Weights input, const NET_INFO& info)
{
    float* ptr = static_cast<float*>(malloc(sizeof(float)*input.count));
    int indir[4]{ 1, 2, 0, 3 };
    int order[5]{ 0, 1, 4, 2, 3};
    int dims[5]{std::get<0>(info), 2, 4, std::get<2>(info), std::get<2>(info)};
    nvinfer1::utils::reshapeWeights(input, dims, order, ptr, 5);
    nvinfer1::utils::transposeSubBuffers(ptr, nvinfer1::DataType::kFLOAT, std::get<0>(info) * 2, std::get<2>(info) * std::get<2>(info), 4);
    int subMatrix = std::get<2>(info) * std::get<2>(info);
    int layerOffset = 8 * subMatrix;
    for (int z = 0; z < std::get<0>(info); ++z) {
        nvinfer1::utils::reorderSubBuffers(ptr + z * layerOffset, indir, 4, subMatrix * sizeof(float));
        nvinfer1::utils::reorderSubBuffers(ptr + z * layerOffset + 4 * subMatrix, indir, 4, subMatrix * sizeof(float));
    }

    return nvinfer1::Weights{input.type, ptr, input.count};
}

// TensorFlow bias parameters for BasicLSTMCell
nvinfer1::Weights convertRNNBias(nvinfer1::Weights input, const NET_INFO& info)
{
    float* ptr = static_cast<float*>(malloc(sizeof(float)*input.count*2));
    std::fill(ptr, ptr + input.count*2, 0);
    const float* iptr = static_cast<const float*>(input.values);
    int indir[4]{ 1, 2, 0, 3 };
    for (int z = 0, y = 0; z < std::get<0>(info); ++z)
        for (int x = 0; x < 4; ++x, ++y)
            std::copy(iptr + y * std::get<2>(info) , iptr + (y + 1) * std::get<2>(info), ptr + (z * 8 + indir[x]) * std::get<2>(info));

    return nvinfer1::Weights{input.type, ptr, input.count*2};
}

// The fully connected weights from tensorflow are transposed compared to the order that tensorRT expects them to be in.
nvinfer1::Weights transposeFCWeights(nvinfer1::Weights input, const NET_INFO& info)
{
    float* ptr = static_cast<float*>(malloc(sizeof(float)*input.count));
    const float* iptr = static_cast<const float*>(input.values);
    assert(input.count == std::get<2>(info) * std::get<5>(info));
    for (int z = 0; z < std::get<2>(info); ++z)
        for (int x = 0; x < std::get<5>(info); ++x)
            ptr[x * std::get<2>(info) + z] = iptr[z * std::get<5>(info) + x];

    return nvinfer1::Weights{input.type, ptr, input.count};
}

int APIToModel(std::map<std::string, nvinfer1::Weights> &weightMap, nvinfer1::IHostMemory** modelStream, const NET_INFO& info, Logger logger)
{
    // create the builder
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);

    // create the model to populate the network, then set the outputs and create an engine
    nvinfer1::INetworkDefinition* network = builder->createNetwork();

    auto data = network->addInput(std::get<6>(info).c_str(), nvinfer1::DataType::kFLOAT, nvinfer1::DimsCHW{ std::get<3>(info), std::get<1>(info), std::get<4>(info)});
    CHECK(data != nullptr);

    auto hiddenIn = network->addInput(std::get<7>(info).c_str(), nvinfer1::DataType::kFLOAT, nvinfer1::DimsCHW{ std::get<0>(info), std::get<1>(info), std::get<2>(info)});
    CHECK(hiddenIn != nullptr);

    auto cellIn = network->addInput(std::get<8>(info).c_str(), nvinfer1::DataType::kFLOAT, nvinfer1::DimsCHW{ std::get<0>(info), std::get<1>(info), std::get<2>(info)});
    CHECK(cellIn != nullptr);

    // Create an RNN layer w/ 2 layers and 512 hidden states
    auto tfwts = weightMap["rnnweight"];
    nvinfer1::Weights rnnwts = convertRNNWeights(tfwts, info);
    auto tfbias = weightMap["rnnbias"];
    nvinfer1::Weights rnnbias = convertRNNBias(tfbias, info);

    auto rnn = network->addRNN(*data, std::get<0>(info), std::get<2>(info), std::get<3>(info),
            nvinfer1::RNNOperation::kLSTM, nvinfer1::RNNInputMode::kLINEAR, nvinfer1::RNNDirection::kUNIDIRECTION, rnnwts, rnnbias);
    CHECK(rnn != nullptr);
    rnn->getOutput(0)->setName("RNN output");
    rnn->setHiddenState(*hiddenIn);
    if (rnn->getOperation() == nvinfer1::RNNOperation::kLSTM)
        rnn->setCellState(*cellIn);
    
    Reshape reshape(std::get<3>(info) * std::get<1>(info) * std::get<2>(info));
    nvinfer1::ITensor *ptr = rnn->getOutput(0);
    auto plugin = network->addPlugin(&ptr, 1, reshape);
    plugin->setName("reshape");

    // Add a second fully connected layer with 50 outputs.
    auto tffcwts = weightMap["rnnfcw"];
    auto wts = transposeFCWeights(tffcwts, info);
    auto bias = weightMap["rnnfcb"];
    auto fc = network->addFullyConnected(*plugin->getOutput(0), std::get<5>(info), wts, bias);
    CHECK(fc != nullptr);
    fc->getOutput(0)->setName("FC output");

    // Add a softmax layer to determine the probability.
    auto prob = network->addSoftMax(*fc->getOutput(0));
    CHECK(prob != nullptr);
    prob->getOutput(0)->setName(std::get<9>(info).c_str());
    network->markOutput(*prob->getOutput(0));
    rnn->getOutput(1)->setName(std::get<10>(info).c_str());
    network->markOutput(*rnn->getOutput(1));
    if (rnn->getOperation() == nvinfer1::RNNOperation::kLSTM) {
        rnn->getOutput(2)->setName(std::get<11>(info).c_str());
        network->markOutput(*rnn->getOutput(2));
    }

    // Build the engine
    builder->setMaxBatchSize(1);
    builder->setMaxWorkspaceSize(1 << 25);

    // Store the transformed weights in the weight map so the memory can be properly released later.
    weightMap["rnnweight2"] = rnnwts;
    weightMap["rnnbias2"] = rnnbias;
    weightMap["rnnfcw2"] = wts;

    auto engine = builder->buildCudaEngine(*network);
    CHECK(engine != nullptr);
    // we don't need the network any more
    network->destroy();

    // serialize the engine, then close everything down
    (*modelStream) = engine->serialize();
    engine->destroy();
    builder->destroy();

    return 0;
}

void stepOnce(float** data, void** buffers, int* sizes, int* indices,
        int numBindings, cudaStream_t& stream, nvinfer1::IExecutionContext &context)
{
    for (int z = 0, w = numBindings/2; z < w; ++z)
        cudaMemcpyAsync(buffers[indices[z]], data[z], sizes[z] * sizeof(float), cudaMemcpyHostToDevice, stream);

    // Execute asynchronously
    context.enqueue(1, buffers, stream, nullptr);

    // DMA the input from the GPU
    for (int z = numBindings/2, w = numBindings; z < w; ++z)
        cudaMemcpyAsync(data[z], buffers[indices[z]], sizes[z] * sizeof(float), cudaMemcpyDeviceToHost, stream);

    // Copy Ct/Ht to the Ct-1/Ht-1 slots.
    cudaMemcpyAsync(data[1], buffers[indices[4]], sizes[1] * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(data[2], buffers[indices[5]], sizes[2] * sizeof(float), cudaMemcpyDeviceToHost, stream);
}

bool doInference(nvinfer1::IExecutionContext& context, const std::string& input, const std::string& expected, std::map<std::string, nvinfer1::Weights>&weightMap, const NET_INFO& info)
{
    const nvinfer1::ICudaEngine& engine = context.getEngine();
    // We have 6 outputs for LSTM, this needs to be changed to 4 for any other RNN type
    static const int numBindings = 6;
    assert(engine.getNbBindings() == numBindings);
    void* buffers[numBindings];
    float* data[numBindings];
    std::fill(buffers, buffers + numBindings, nullptr);
    std::fill(data, data + numBindings, nullptr);
    const char* names[numBindings] = {std::get<6>(info).c_str(), std::get<7>(info).c_str(), std::get<8>(info).c_str(),
                                    std::get<9>(info).c_str(), std::get<10>(info).c_str(), std::get<11>(info).c_str() };
    int indices[numBindings];
    std::fill(indices, indices + numBindings, -1);
    int sizes[numBindings] = { std::get<3>(info) * std::get<1>(info) * std::get<4>(info),
                                std::get<0>(info) * std::get<1>(info) * std::get<2>(info),
                                std::get<0>(info) * std::get<1>(info) * std::get<2>(info),
                                std::get<5>(info),
                                std::get<0>(info) * std::get<1>(info) * std::get<2>(info),
                                std::get<0>(info) * std::get<1>(info) * std::get<2>(info) };

    for (int x = 0; x < numBindings; ++x) {
        // In order to bind the buffers, we need to know the names of the input and output tensors.
        // note that indices are guaranteed to be less than IEngine::getNbBindings()
        indices[x] = engine.getBindingIndex(names[x]);
        if (indices[x] == -1) continue;
        // create GPU buffers and a stream
        assert(indices[x] < numBindings);
        cudaMalloc(&buffers[indices[x]], sizes[x] * sizeof(float));
        data[x] = new float[sizes[x]];
    }
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    // Initialize input/hidden/cell state to zero
    for (int x = 0; x < numBindings; ++x) std::fill(data[x], data[x] + sizes[x], 0.0f);

    auto embed = weightMap["embed"];
    std::string genstr;
    assert(std::get<1>(info) == 1 && "This code assumes batch size is equal to 1.");
    // Seed the RNN with the input.
    for (auto &a : input) {
        std::copy(reinterpret_cast<const float*>(embed.values) + char_to_id[a]*std::get<4>(info),
                reinterpret_cast<const float*>(embed.values) + char_to_id[a]*std::get<4>(info) + std::get<4>(info),
                data[0]);
        stepOnce(data, buffers, sizes, indices, 6, stream, context);
        cudaStreamSynchronize(stream);
        genstr.push_back(a);
    }
    // Now that we have gone through the initial sequence, lets make sure that we get the sequence out that
    // we are expecting.
    for (size_t x = 0, y = expected.size(); x < y; ++x) {
        std::copy(reinterpret_cast<const float*>(embed.values) + char_to_id[*genstr.rbegin()]*std::get<4>(info),
                reinterpret_cast<const float*>(embed.values) + char_to_id[*genstr.rbegin()]*std::get<4>(info) + std::get<4>(info),
                data[0]);

        stepOnce(data, buffers, sizes, indices, 6, stream, context);
        cudaStreamSynchronize(stream);

		float* probabilities = reinterpret_cast<float*>(data[indices[3]]);
		ptrdiff_t idx = std::max_element(probabilities, probabilities + sizes[3]) - probabilities;
        genstr.push_back(id_to_char[idx]);
    }

    fprintf(stdout, "Received: %s\n", genstr.c_str() + input.size());

    // release the stream and the buffers
    cudaStreamDestroy(stream);
    for (int x = 0; x < numBindings; ++x) {
        cudaFree(buffers[indices[x]]);
        if (data[x]) delete [] data[x];
    }

    return genstr == (input + expected);
}

} // namespace

int test_charrnn()
{
    const NET_INFO info(2, 1, 512, 1, 512, 50, "data", "hiddenIn", "cellIn", "prob", "hiddenOut", "cellOut");
    Logger logger; // multiple instances of IRuntime and/or IBuilder must all use the same logger
    // create a model using the API directly and serialize it to a stream
    nvinfer1::IHostMemory* modelStream{ nullptr };

    std::map<std::string, nvinfer1::Weights> weightMap = loadWeights("models/char-rnn.wts");
    APIToModel(weightMap, &modelStream, info, logger);

    const std::vector<std::string> in_strs {"customer serv", "business plans", "help", "slightly under", "market",
                            "holiday cards", "bring it", "what time", "the owner thinks", "money can be use"};
    const std::vector<std::string> out_strs { "es and the", " to be a", "en and", "iting the company", "ing and",
                        " the company", " company said it will", "d and the company", "ist with the", "d to be a"};
    CHECK(in_strs.size() == out_strs.size());

    PluginFactory pluginFactory;

    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);
    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(modelStream->data(), modelStream->size(), &pluginFactory);
    nvinfer1::IExecutionContext* context = engine->createExecutionContext();

    for (int num = 0; num < in_strs.size(); ++num) {
        bool pass {false};
        fprintf(stdout, "RNN Warmup: %s, Expect: %s\n", in_strs[num].c_str(), out_strs[num].c_str());
        pass = doInference(*context, in_strs[num], out_strs[num], weightMap, info);
        if (!pass) fprintf(stderr, "Failure!\n");
    }

    if (modelStream) modelStream->destroy();
    for (auto& mem : weightMap) {
        free((void*)(mem.second.values));
    }

    // destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();
    pluginFactory.destroyPlugin();

    return 0;
}
