#include "Validator.h"
#include "LayerConfiguration.h"

using namespace GNA;

ConfigurationBuffer::ConfigurationBuffer(gna_buffer_type typeIn, void* address) :
    InOutBuffer{address},
    type{typeIn}
{
    Expect::NotNull(buffer);
}
