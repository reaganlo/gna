#include <cstring>
#include <cstdlib>
#include <cstdio>
#include "gna-api.h"

void print_outputs(
    int32_t *outputs,
    uint32_t nRows,
    uint32_t nColumns
)
{
    printf("\nOutputs:\n");
    for(int i = 0; i < nRows; ++i)
    {
        for(int j = 0; j < nColumns; ++j)
        {
            printf("%d\t", outputs[i*nColumns + j]);
        }
        putchar('\n');
    }
    putchar('\n');
}

int wmain(int argc, wchar_t *argv[])
{
    intel_gna_status_t status = GNA_SUCCESS; // for simplicity sake status codes are not examined after api functions calls
                                             // it is highly recommended to inspect the status every time, and act accordingly
    // open the device
    gna_device_id gna_handle;
    GnaDeviceOpen(1, &gna_handle);


    intel_nnet_type_t nnet;  // main neural network container
    nnet.nGroup = 4;         // grouping factor (1-8), specifies how many input vectors are simultaneously run through the nnet
    nnet.nLayers = 1;        // number of hidden layers, using 1 for simplicity sake
    nnet.pLayers = (intel_nnet_layer_t*)calloc(nnet.nLayers, sizeof(intel_nnet_layer_t));   // container for layer definitions

    int16_t weights[8 * 16] = {                                          // sample weight matrix (8 rows, 16 cols)
        -6, -2, -1, -1, -2,  9,  6,  5,  2,  4, -1,  5, -2, -4,  0,  9,  // in case of affine layer this is the left operand of matrix mul
        -8,  8, -4,  6,  5,  3, -7, -9,  7,  0, -4, -1,  1,  7,  6, -6,  // in this sample the numbers are random and meaningless
         2, -8,  6,  5, -1, -2,  7,  5, -1,  4,  8,  7, -9, -1,  7,  1,
         0, -2,  1,  0,  6, -6,  7,  4, -6,  0,  3, -2,  1,  8, -6, -2,
        -6, -3,  4, -2, -8, -6,  6,  5,  6, -9, -5, -2, -5, -8, -6, -2,
        -7,  0,  6, -3, -1, -6,  4,  1, -4, -5, -3,  7,  9, -9,  9,  9,
         0, -2,  6, -3,  5, -2, -1, -3, -5,  7,  6,  6, -8,  0, -4,  9,
         2,  7, -8, -7,  8, -6, -6,  1,  7, -4, -4,  9, -6, -6,  5, -7
    };

    int16_t inputs[16 * 4] = {      // sample input matrix (16 rows, 4 cols), consists of 4 input vectors (grouping of 4 is used)
        -5,  9, -7,  4,             // in case of affine layer this is the right operand of matrix mul
         5, -4, -7,  4,             // in this sample the numbers are random and meaningless
         0,  7,  1, -7,
         1,  6,  7,  9,
         2, -4,  9,  8,
        -5, -1,  2,  9,
        -8, -8,  8,  1,
        -7,  2, -1, -1,
        -9, -5, -8,  5,
         0, -1,  3,  9,
         0,  8,  1, -2,
        -9,  8,  0, -7,
        -9, -8, -1, -4,
        -3, -7, -2,  3,
        -8,  0,  1,  3,
        -4, -6, -8, -2
    };

    int32_t biases[8] = {      // sample bias vector, will get added to each of the four output vectors
         5,                    // in this sample the numbers are random and meaningless
         4,
        -2,
         5,
        -7,
        -5,
         4,
        -1
    };

    int buf_size_weights     = ALIGN64(sizeof(weights)); // note that buffer alignment to 64-bytes is required by GNA HW
    int buf_size_inputs      = ALIGN64(sizeof(inputs));
    int buf_size_biases      = ALIGN64(sizeof(biases));
    int buf_size_outputs     = ALIGN64(8 * 4 * 4);       // (4 out vectors, 8 elems in each one, 4-byte elems)
    int buf_size_tmp_outputs = ALIGN64(8 * 4 * 4);       // (4 out vectors, 8 elems in each one, 4-byte elems)

    // prepare params for GNAAlloc
    uint32_t bytes_requested = buf_size_weights + buf_size_inputs + buf_size_biases + buf_size_outputs + buf_size_tmp_outputs;
    uint32_t bytes_granted;

    // call GNAAlloc (obtains pinned memory shared with the device)
    uint8_t *pinned_mem_ptr = (uint8_t*)GnaAlloc(gna_handle, bytes_requested, 1, 0, &bytes_granted);

    int16_t *pinned_weights = (int16_t*)pinned_mem_ptr;
    memcpy(pinned_weights, weights, sizeof(weights));   // puts the weights into the pinned memory
    pinned_mem_ptr += buf_size_weights;                 // fast-forwards current pinned memory pointer to the next free block

    int16_t *pinned_inputs = (int16_t*)pinned_mem_ptr;
    memcpy(pinned_inputs, inputs, sizeof(inputs));      // puts the inputs into the pinned memory
    pinned_mem_ptr += buf_size_inputs;                  // fast-forwards current pinned memory pointer to the next free block

    int32_t *pinned_biases = (int32_t*)pinned_mem_ptr;
    memcpy(pinned_biases, biases, sizeof(biases));      // puts the biases into the pinned memory
    pinned_mem_ptr += buf_size_biases;                  // fast-forwards current pinned memory pointer to the next free block

    int16_t *pinned_outputs = (int16_t*)pinned_mem_ptr;
    pinned_mem_ptr += buf_size_outputs;                 // fast-forwards the current pinned memory pointer by the space needed for outputs

    int32_t *pinned_tmp_outputs = (int32_t*)pinned_mem_ptr;      // the last free block will be used for GNA's scratch pad

    intel_affine_func_t affine_func;       // parameters needed for the affine transformation are held here
    affine_func.nBytesPerWeight = 2;
    affine_func.nBytesPerBias = 4;
    affine_func.pWeights = pinned_weights;
    affine_func.pBiases = pinned_biases;

    intel_pwl_func_t pwl;                  // no piecewise linear activation function used in this simple example
    pwl.nSegments = 0;
    pwl.pSegments = NULL;

    intel_affine_layer_t affine_layer;     // affine layer combines the affine transformation and activation function
    affine_layer.affine = affine_func;
    affine_layer.pwl = pwl;

    intel_nnet_layer_t nnet_layer;         // contains the definition of a single layer
    nnet_layer.nInputColumns = nnet.nGroup;
    nnet_layer.nInputRows = 16;
    nnet_layer.nOutputColumns = nnet.nGroup;
    nnet_layer.nOutputRows = 8;
    nnet_layer.nBytesPerInput = 2;
    nnet_layer.nBytesPerOutput = 4;             // 4 bytes since we are not using PWL (would be 2 bytes otherwise)
    nnet_layer.nBytesPerIntermediateOutput = 4; // this is always 4 bytes
    nnet_layer.type = INTEL_INPUT_OUTPUT;
    nnet_layer.nLayerKind = INTEL_AFFINE;
    nnet_layer.pLayerStruct = &affine_layer;
    nnet_layer.pInputs = nullptr;
    nnet_layer.pOutputsIntermediate = nullptr;
    nnet_layer.pOutputs = nullptr;

    memcpy(nnet.pLayers, &nnet_layer, sizeof(nnet_layer));   // puts the layer into the main network container
                                                             // if there was another layer to add, it would get copied to nnet.pLayers + 1

    gna_model_id model_id;
    GnaModelCreate(gna_handle, &nnet, &model_id);

    gna_request_cfg_id config_id;
    GnaModelRequestConfigAdd(model_id, &config_id);
    GnaRequestConfigBufferAdd(config_id, GNA_IN, 0, pinned_inputs);
    GnaRequestConfigBufferAdd(config_id, GNA_OUT, 0, pinned_outputs);

    // calculate on GNA HW (non-blocking call)
    gna_request_id request_id;     // this gets filled with the actual id later on
    status = GnaRequestEnqueue(config_id, GNA_GENERIC, &request_id);

    /**************************************************************************************************
     * Offload effect: other calculations can be done on CPU here, while nnet decoding runs on GNA HW *
     **************************************************************************************************/

    // wait for HW calculations (blocks until the results are ready)
    gna_timeout timeout = 1000;
    status = GnaRequestWait(request_id, timeout);     // after this call, outputs can be inspected under nnet.pLayers->pOutputs

    print_outputs((int32_t*)pinned_outputs, nnet.pLayers->nOutputRows, nnet.pLayers->nOutputColumns);

                                                      // -177  -85   29   28
    // free the pinned memory                         //   96 -173   25  252
    status = GnaFree(gna_handle);                     // -160  274  157  -29
                                                      //   48  -60  158  -29
    // free heap allocations                          //   26   -2  -44 -251
    free(nnet.pLayers);                               // -173  -70   -1 -323
                                                      //   99  144   38  -63
    // close the device                               //   20   56 -103   10
    status = GnaDeviceClose(gna_handle);

    return 0;
}
