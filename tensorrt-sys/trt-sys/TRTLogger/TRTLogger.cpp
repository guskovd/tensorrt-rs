//
// Created by mason on 8/25/19.
//
#include <cstdio>
#include "NvInfer.h"
#include "TRTLoggerInternal.hpp"

void get_tensorrt_version(char *string) {
    sprintf(string, "%d.%d.%d", NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR, NV_TENSORRT_PATCH);
}

Logger_t *create_logger() {
    return new Logger();
}

void delete_logger(Logger_t *logger) {
    if (logger == nullptr)
        return;

    delete logger;
}

void log_error(Logger_t *logger, char *err) {
    if (logger == nullptr)
        return;
    auto &l = logger->internal_logger;
    l->log(nvinfer1::ILogger::Severity::kWARNING, err);
}

nvinfer1::ILogger &Logger::getLogger() const {
    return *this->internal_logger;
}