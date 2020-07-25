#pragma clang diagnostic push
#pragma ide diagnostic ignored "modernize-deprecated-headers"
//
// Created by mason on 9/17/19.
//

#ifndef LIBTRT_TRTCONTEXT_H
#define LIBTRT_TRTCONTEXT_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct Context;
typedef struct Context Context_t;

void destroy_excecution_context(Context_t* execution_context);

void context_set_name(Context_t* execution_context, const char *name);
const char* context_get_name(Context_t *execution_context);

void execute(Context_t* execution_context, const float* input_data, const size_t input_data_size, const int input_index, float *output_data, const size_t output_data_size, const int output_index);
void execute_bindings_v2(Context_t* execution_context, void** bindings);

bool set_optimization_profile(Context_t* execution_context, int profile_index);
bool set_binding_shape_dims2(Context_t* execution_context, int binding_index, int d0, int d1);

#ifdef __cplusplus
};
#endif


#endif //LIBTRT_TRTCONTEXT_H

#pragma clang diagnostic pop
