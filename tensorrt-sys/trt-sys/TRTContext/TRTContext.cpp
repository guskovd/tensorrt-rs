//
// Created by mason on 9/17/19.
//
#include <cstdlib>
#include <memory>
#include <iostream> 
#include <cuda_runtime.h>
#include "NvInfer.h"
#include "TRTContextInternal.hpp"
#include "../TRTUtils.hpp"

struct Context {
    using IExecutionContextPtr = std::unique_ptr<nvinfer1::IExecutionContext, TRTDeleter<nvinfer1::IExecutionContext>>;
    IExecutionContextPtr internal_context;

    explicit Context(nvinfer1::IExecutionContext* executionContext) {
        internal_context = IExecutionContextPtr(executionContext);
    }

};

Context_t* create_execution_context(nvinfer1::IExecutionContext* execution_context) {
    return new Context(execution_context);
}

void destroy_excecution_context (Context_t* execution_context) {
    if (execution_context == nullptr)
        return;

    delete execution_context;
}

void context_set_name(Context_t *execution_context, const char * name) {
    if (execution_context == nullptr)
        return;

    execution_context->internal_context->setName(name);
}

const char* context_get_name(Context_t *execution_context) {
    if (execution_context == nullptr)
        return "";

    return execution_context->internal_context->getName();
}

void execute(Context_t* execution_context, const float* input_data, const size_t input_data_size, const int input_index, float *output_data, const size_t output_data_size, const int output_index) {
    if (execution_context == nullptr)
        return;
    auto& context = execution_context->internal_context;

    void* buffers[2];
    cudaMalloc(&buffers[0], input_data_size);
    cudaMalloc(&buffers[1], output_data_size);
    

    cudaMemcpy(buffers[input_index], input_data, input_data_size, cudaMemcpyHostToDevice);
    context->execute(1, &buffers[input_index]);
    cudaMemcpy(output_data, buffers[output_index], output_data_size, cudaMemcpyDeviceToHost);

    cudaFree(&buffers[0]);
    cudaFree(&buffers[1]);
}

void execute_v2(Context_t* execution_context, const float* input_data, const size_t input_data_size, const int input_index, float *output_data, const size_t output_data_size, const int output_index) {
    if (execution_context == nullptr)
        return;
    auto& context = execution_context->internal_context;

    void* buffers[2];
    cudaMalloc(&buffers[0], input_data_size);
    cudaMalloc(&buffers[1], output_data_size);

    cudaMemcpy(buffers[input_index], input_data, input_data_size, cudaMemcpyHostToDevice);
    context->executeV2(buffers);
    cudaMemcpy(output_data, buffers[output_index], output_data_size, cudaMemcpyDeviceToHost);

    cudaFree(&buffers[0]);
    cudaFree(&buffers[1]);
}

void execute_bindings_v2(Context_t* execution_context, void** bindings) {
    if (execution_context == nullptr)
        return;
    auto& context = execution_context->internal_context;
    context->executeV2(bindings);
}

bool set_optimization_profile(Context_t* execution_context, int profile_index) {
    return execution_context->internal_context->setOptimizationProfile(profile_index);
}

bool set_binding_shape_dims2(Context_t* execution_context, int binding_index, int d0, int d1) {
    return execution_context->internal_context->setBindingDimensions(binding_index, nvinfer1::Dims2(d0, d1));
}
