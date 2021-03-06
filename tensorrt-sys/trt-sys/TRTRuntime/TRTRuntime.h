//
// Created by mason on 8/25/19.
//

#ifndef TENSRORT_SYS_TRTRUNTIME_H
#define TENSRORT_SYS_TRTRUNTIME_H

#include "../TRTLogger/TRTLogger.h"
#include "../TRTCudaEngine/TRTCudaEngine.h"

#ifdef __cplusplus
extern "C" {
#endif

struct Runtime;
typedef struct Runtime Runtime_t;

Runtime_t* create_infer_runtime(Logger_t* logger);
Engine_t* deserialize_cuda_engine(Runtime_t* runtime, const void* blob, unsigned long long size);
void destroy_infer_runtime(Runtime_t* runtime);

#ifdef __cplusplus
};
#endif

#endif //TENSRORT_SYS_TRTRUNTIME_H
