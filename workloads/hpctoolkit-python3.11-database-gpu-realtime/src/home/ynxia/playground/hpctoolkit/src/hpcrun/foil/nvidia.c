// SPDX-FileCopyrightText: Contributors to the HPCToolkit Project
//
// SPDX-License-Identifier: BSD-3-Clause

#define _GNU_SOURCE

#include "nvidia.h"

#include "../hpcrun-sonames.h"
#include "../messages/messages.h"
#include "nvidia-private.h"

#include <assert.h>
#include <dlfcn.h>
#include <stdbool.h>
#include <stdlib.h>
#include <threads.h>

static const struct hpcrun_foil_appdispatch_nvidia* dispatch_var = NULL;

static void init_dispatch() {
  void* handle = dlmopen(LM_ID_BASE, HPCRUN_DLOPEN_NVIDIA_SO, RTLD_NOW | RTLD_DEEPBIND);
  if (handle == NULL) {
    EEMSG("CUDA support failed to load, some functionality may be unavailable: %s",
          dlerror());
    return;
  }
  dispatch_var = dlsym(handle, "hpcrun_dispatch_nvidia");
  if (dispatch_var == NULL) {
    EEMSG("Inconsistent " HPCRUN_DLOPEN_NVIDIA_SO " found, CUDA support is disabled.");
    return;
  }
}

static const struct hpcrun_foil_appdispatch_nvidia* dispatch() {
  static once_flag once = ONCE_FLAG_INIT;
  call_once(&once, init_dispatch);
  return dispatch_var;
}

CUptiResult f_cuptiActivityEnable(CUpti_ActivityKind kind) {
  const struct hpcrun_foil_appdispatch_nvidia* d = dispatch();
  if (d == NULL)
    return CUPTI_ERROR_NOT_INITIALIZED;
  return d->cuptiActivityEnable(kind);
}

CUptiResult f_cuptiActivityDisable(CUpti_ActivityKind kind) {
  const struct hpcrun_foil_appdispatch_nvidia* d = dispatch();
  if (d == NULL)
    return CUPTI_ERROR_NOT_INITIALIZED;
  return d->cuptiActivityDisable(kind);
}

CUptiResult f_cuptiActivityEnableContext(CUcontext context, CUpti_ActivityKind kind) {
  const struct hpcrun_foil_appdispatch_nvidia* d = dispatch();
  if (d == NULL)
    return CUPTI_ERROR_NOT_INITIALIZED;
  return d->cuptiActivityEnableContext(context, kind);
}

CUptiResult f_cuptiActivityDisableContext(CUcontext context, CUpti_ActivityKind kind) {
  const struct hpcrun_foil_appdispatch_nvidia* d = dispatch();
  if (d == NULL)
    return CUPTI_ERROR_NOT_INITIALIZED;
  return d->cuptiActivityDisableContext(context, kind);
}

CUptiResult f_cuptiActivityConfigurePCSampling(CUcontext ctx,
                                               CUpti_ActivityPCSamplingConfig* config) {
  const struct hpcrun_foil_appdispatch_nvidia* d = dispatch();
  if (d == NULL)
    return CUPTI_ERROR_NOT_INITIALIZED;
  return d->cuptiActivityConfigurePCSampling(ctx, config);
}

CUptiResult f_cuptiActivityRegisterCallbacks(
    CUpti_BuffersCallbackRequestFunc funcBufferRequested,
    CUpti_BuffersCallbackCompleteFunc funcBufferCompleted) {
  const struct hpcrun_foil_appdispatch_nvidia* d = dispatch();
  if (d == NULL)
    return CUPTI_ERROR_NOT_INITIALIZED;
  return d->cuptiActivityRegisterCallbacks(funcBufferRequested, funcBufferCompleted);
}

CUptiResult f_cuptiActivityPushExternalCorrelationId(CUpti_ExternalCorrelationKind kind,
                                                     uint64_t id) {
  const struct hpcrun_foil_appdispatch_nvidia* d = dispatch();
  if (d == NULL)
    return CUPTI_ERROR_NOT_INITIALIZED;
  return d->cuptiActivityPushExternalCorrelationId(kind, id);
}

CUptiResult f_cuptiActivityPopExternalCorrelationId(CUpti_ExternalCorrelationKind kind,
                                                    uint64_t* lastId) {
  const struct hpcrun_foil_appdispatch_nvidia* d = dispatch();
  if (d == NULL)
    return CUPTI_ERROR_NOT_INITIALIZED;
  return d->cuptiActivityPopExternalCorrelationId(kind, lastId);
}

CUptiResult f_cuptiActivityGetNextRecord(uint8_t* buffer, size_t validBufferSizeBytes,
                                         CUpti_Activity** record) {
  const struct hpcrun_foil_appdispatch_nvidia* d = dispatch();
  if (d == NULL)
    return CUPTI_ERROR_NOT_INITIALIZED;
  return d->cuptiActivityGetNextRecord(buffer, validBufferSizeBytes, record);
}

CUptiResult f_cuptiActivityGetNumDroppedRecords(CUcontext context, uint32_t streamId,
                                                size_t* dropped) {
  const struct hpcrun_foil_appdispatch_nvidia* d = dispatch();
  if (d == NULL)
    return CUPTI_ERROR_NOT_INITIALIZED;
  return d->cuptiActivityGetNumDroppedRecords(context, streamId, dropped);
}

CUptiResult f_cuptiActivitySetAttribute(CUpti_ActivityAttribute attribute,
                                        size_t* value_size, void* value) {
  const struct hpcrun_foil_appdispatch_nvidia* d = dispatch();
  if (d == NULL)
    return CUPTI_ERROR_NOT_INITIALIZED;
  return d->cuptiActivitySetAttribute(attribute, value_size, value);
}

CUptiResult f_cuptiActivityFlushAll(uint32_t flag) {
  const struct hpcrun_foil_appdispatch_nvidia* d = dispatch();
  if (d == NULL)
    return CUPTI_ERROR_NOT_INITIALIZED;
  return d->cuptiActivityFlushAll(flag);
}

CUptiResult f_cuptiGetTimestamp(uint64_t* timestamp) {
  const struct hpcrun_foil_appdispatch_nvidia* d = dispatch();
  if (d == NULL)
    return CUPTI_ERROR_NOT_INITIALIZED;
  return d->cuptiGetTimestamp(timestamp);
}

CUptiResult f_cuptiEnableDomain(uint32_t enable, CUpti_SubscriberHandle subscriber,
                                CUpti_CallbackDomain domain) {
  const struct hpcrun_foil_appdispatch_nvidia* d = dispatch();
  if (d == NULL)
    return CUPTI_ERROR_NOT_INITIALIZED;
  return d->cuptiEnableDomain(enable, subscriber, domain);
}

CUptiResult f_cuptiFinalize() {
  const struct hpcrun_foil_appdispatch_nvidia* d = dispatch();
  if (d == NULL)
    return CUPTI_ERROR_NOT_INITIALIZED;
  return d->cuptiFinalize();
}

CUptiResult f_cuptiGetResultString(CUptiResult result, const char** str) {
  const struct hpcrun_foil_appdispatch_nvidia* d = dispatch();
  if (d == NULL)
    return CUPTI_ERROR_NOT_INITIALIZED;
  return d->cuptiGetResultString(result, str);
}

CUptiResult f_cuptiSubscribe(CUpti_SubscriberHandle* subscriber,
                             CUpti_CallbackFunc callback, void* userdata) {
  const struct hpcrun_foil_appdispatch_nvidia* d = dispatch();
  if (d == NULL)
    return CUPTI_ERROR_NOT_INITIALIZED;
  return d->cuptiSubscribe(subscriber, callback, userdata);
}

CUptiResult f_cuptiEnableCallback(uint32_t enable, CUpti_SubscriberHandle subscriber,
                                  CUpti_CallbackDomain domain, CUpti_CallbackId cbid) {
  const struct hpcrun_foil_appdispatch_nvidia* d = dispatch();
  if (d == NULL)
    return CUPTI_ERROR_NOT_INITIALIZED;
  return d->cuptiEnableCallback(enable, subscriber, domain, cbid);
}

CUptiResult f_cuptiUnsubscribe(CUpti_SubscriberHandle subscriber) {
  const struct hpcrun_foil_appdispatch_nvidia* d = dispatch();
  if (d == NULL)
    return CUPTI_ERROR_NOT_INITIALIZED;
  return d->cuptiUnsubscribe(subscriber);
}

CUresult f_cuDeviceGetAttribute(int* pi, CUdevice_attribute attrib, CUdevice dev) {
  const struct hpcrun_foil_appdispatch_nvidia* d = dispatch();
  if (d == NULL)
    return CUDA_ERROR_NO_DEVICE;
  return d->cuDeviceGetAttribute(pi, attrib, dev);
}

CUresult f_cuCtxGetCurrent(CUcontext* ctx) {
  const struct hpcrun_foil_appdispatch_nvidia* d = dispatch();
  if (d == NULL)
    return CUDA_ERROR_NO_DEVICE;
  return d->cuCtxGetCurrent(ctx);
}

CUresult f_cuFuncGetModule(CUmodule* hmod, CUfunction function) {
  const struct hpcrun_foil_appdispatch_nvidia* d = dispatch();
  if (d == NULL)
    return CUDA_ERROR_NO_DEVICE;
  return d->cuFuncGetModule(hmod, function);
}

CUresult f_cuDriverGetVersion(int* version) {
  const struct hpcrun_foil_appdispatch_nvidia* d = dispatch();
  if (d == NULL)
    return CUDA_ERROR_NO_DEVICE;
  return d->cuDriverGetVersion(version);
}

cudaError_t f_cudaGetDevice(int* device_id) {
  const struct hpcrun_foil_appdispatch_nvidia* d = dispatch();
  if (d == NULL)
    return cudaErrorNoDevice;
  return d->cudaGetDevice(device_id);
}

cudaError_t f_cudaRuntimeGetVersion(int* runtimeVersion) {
  const struct hpcrun_foil_appdispatch_nvidia* d = dispatch();
  if (d == NULL)
    return cudaErrorNoDevice;
  return d->cudaRuntimeGetVersion(runtimeVersion);
}

cudaError_t f_cudaDeviceSynchronize() {
  const struct hpcrun_foil_appdispatch_nvidia* d = dispatch();
  if (d == NULL)
    return cudaErrorNoDevice;
  return d->cudaDeviceSynchronize();
}

cudaError_t f_cudaMemcpy(void* dst, const void* src, size_t count,
                         enum cudaMemcpyKind kind) {
  const struct hpcrun_foil_appdispatch_nvidia* d = dispatch();
  if (d == NULL)
    return cudaErrorNoDevice;
  return d->cudaMemcpy(dst, src, count, kind);
}
