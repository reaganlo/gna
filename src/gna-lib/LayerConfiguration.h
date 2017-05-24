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

struct LayerConfiguration
{
    std::unique_ptr<ActiveList> ActiveList;
    std::unique_ptr<ConfigurationBuffer> InputBuffer;
    std::unique_ptr<ConfigurationBuffer> OutputBuffer;

    std::unique_ptr<AffineConfig> affineConfig;
    std::unique_ptr<AffineConfigAl> activeListConfig;
    std::unique_ptr<PwlOutputConfig> pwlOutputConfig;
    std::unique_ptr<RecurrentConfig> recurrentConfig;
    std::unique_ptr<ConvolutionConfig> convolutionConfig;
    std::unique_ptr<TransposeConfig> transposeConfig;
    std::unique_ptr<CopyConfig> copyConfig;
    std::unique_ptr<GmmConfig> gmmConfig;
};
}
