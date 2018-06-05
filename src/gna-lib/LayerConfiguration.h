#pragma once

#include <vector>

#include "ActiveList.h"
#include "Address.h"
#include "KernelArguments.h"

namespace GNA
{
struct ConfigurationBuffer : public InOutBuffer
{
    ConfigurationBuffer(gna_buffer_type type, void *address);

    ConfigurationBuffer(ConfigurationBuffer &&) = default;

    ConfigurationBuffer() = delete;
    ConfigurationBuffer(const ConfigurationBuffer &) = delete;
    ConfigurationBuffer& operator=(const ConfigurationBuffer&) = delete;

    gna_buffer_type type;
};

struct KernelConfigs
{
    std::unique_ptr<const AffineConfig> Affine;
    std::unique_ptr<RecurrentConfig> Recurrent;
    std::unique_ptr<const ConvolutionConfig> Convolution;
    std::unique_ptr<TransposeConfig> Transpose;
    std::unique_ptr<CopyConfig> Copy;
    std::unique_ptr<GmmConfig> Gmm;
    std::unique_ptr<PwlOutputConfig> PwlOutput;
};

struct LayerConfiguration
{
    std::unique_ptr<ActiveList> ActList;
    std::unique_ptr<ConfigurationBuffer> InputBuffer;
    std::unique_ptr<ConfigurationBuffer> OutputBuffer;
    KernelConfigs Configs;
};
}
