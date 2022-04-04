/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/api/lib/api_custom_impl.h"

#include "paddle/phi/api/lib/api_gen_utils.h"
#include "paddle/phi/api/lib/data_transform.h"
#include "paddle/phi/api/lib/kernel_dispatch.h"
#include "paddle/phi/api/lib/utils/storage.h"
#include "paddle/phi/core/compat/convert_utils.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/meta_tensor.h"
#include "paddle/phi/infermeta/backward.h"
#include "paddle/phi/infermeta/binary.h"
#include "paddle/phi/infermeta/multiary.h"
#include "paddle/phi/infermeta/nullary.h"
#include "paddle/phi/infermeta/unary.h"

#include "glog/logging.h"

namespace paddle {
namespace experimental {

// TODO(chenweihang):  the original sum grad op can support higher-level
// differentiation,
// but if we use this impl, it will not support. We need to be able to reuse
// the autograd API here, which is not yet implemented
// TODO(chenweihang): we should support call generated api in custom api impl
std::vector<Tensor> add_n_grad_impl(const std::vector<Tensor>& x,
                                    const Tensor& out_grad) {
  auto kernel_key_set = ParseKernelKeyByInputArgs(out_grad);
  auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();

  Backend kernel_backend = kernel_key.backend();
  DataLayout kernel_layout = kernel_key.layout();
  DataType kernel_data_type = kernel_key.dtype();

  auto kernel = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "scale", {kernel_backend, kernel_layout, kernel_data_type});
  VLOG(6) << "add_n_grad API kernel key: [" << kernel_backend << ", "
          << kernel_layout << ", " << kernel_data_type << "]";
  VLOG(6) << "add_n_grad API kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_backend);

  auto dense_out_grad = PrepareData(out_grad, kernel.InputAt(0), {});

  size_t out_number = x.size();
  std::vector<Tensor> x_grad;
  auto dense_x_grad = SetKernelOutput(out_number, kernel_backend, &x_grad);

  using kernel_signature = void (*)(const platform::DeviceContext&,
                                    const phi::DenseTensor&,
                                    const phi::Scalar&,
                                    float,
                                    bool,
                                    phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();

  for (auto* dense_x_grad_t : dense_x_grad) {
    phi::MetaTensor meta_out(dense_x_grad_t);
    phi::UnchangedInferMeta(MakeMetaTensor(*dense_out_grad), &meta_out);
    (*kernel_fn)(
        *dev_ctx, *dense_out_grad, phi::Scalar(1.0), 0.0, true, dense_x_grad_t);
  }

  return x_grad;
}

Tensor copy_to_impl(const Tensor& x, Place place, bool blocking) {
  auto kernel_key_set = ParseKernelKeyByInputArgs(x);
  kernel_key_set.backend_set =
      kernel_key_set.backend_set | BackendSet(phi::TransToPhiBackend(place));
  auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
  auto kernel = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "copy", kernel_key);

  VLOG(6) << "copy API kernel key: " << kernel_key;
  VLOG(6) << "copy API kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_key.backend());

  auto dense_x = TensorToDenseTensor(x);

  Tensor out;
  auto kernel_out = SetKernelOutput(kernel_key.backend(), &out);
  phi::MetaTensor meta_out(kernel_out);
  phi::UnchangedInferMeta(*dense_x, &meta_out);

  using kernel_signature = void (*)(const platform::DeviceContext&,
                                    const phi::DenseTensor&,
                                    phi::Place,
                                    bool,
                                    phi::DenseTensor*);

  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  (*kernel_fn)(*dev_ctx, *dense_x, place, blocking, kernel_out);

  return out;
}

std::vector<Tensor> split_impl(const Tensor& x,
                               const IntArray& num_or_sections,
                               const Scalar& axis) {
  auto kernel_key_set = ParseKernelKeyByInputArgs(x);
  auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();

  Backend kernel_backend = kernel_key.backend();
  DataLayout kernel_layout = kernel_key.layout();
  DataType kernel_data_type = kernel_key.dtype();

  auto kernel = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "split", {kernel_backend, kernel_layout, kernel_data_type});
  VLOG(6) << "split API kernel key: [" << kernel_backend << ", "
          << kernel_layout << ", " << kernel_data_type << "]";
  VLOG(6) << "split API kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_backend);

  auto dense_x = PrepareData(x, kernel.InputAt(0), {});

  // Calculate the number of out tensors
  size_t out_number;
  if (num_or_sections.GetData().size() == 1) {
    out_number = num_or_sections.GetData()[0];
  } else {
    out_number = num_or_sections.GetData().size();
  }

  std::vector<Tensor> out;
  auto dense_outs = SetKernelOutput(out_number, kernel_backend, &out);
  std::vector<phi::MetaTensor> meta_outs;
  meta_outs.reserve(out_number);
  std::vector<phi::MetaTensor*> meta_out_ptrs;
  meta_out_ptrs.reserve(out_number);
  for (size_t i = 0; i < out_number; ++i) {
    meta_outs.push_back(dense_outs[i]);
    meta_out_ptrs.push_back(&meta_outs.back());
  }

  phi::SplitInferMeta(
      MakeMetaTensor(*dense_x), num_or_sections, axis, meta_out_ptrs);

  using kernel_signature = void (*)(const platform::DeviceContext&,
                                    const phi::DenseTensor&,
                                    const phi::IntArray&,
                                    const phi::Scalar&,
                                    std::vector<phi::DenseTensor*>&);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  (*kernel_fn)(*dev_ctx,
               *dense_x,
               phi::IntArray(num_or_sections),
               phi::Scalar(axis),
               dense_outs);

  return out;
}

std::vector<Tensor> concat_grad_impl(const std::vector<Tensor>& x,
                                     const Tensor& out_grad,
                                     const Scalar& axis) {
  auto kernel_key_set = ParseKernelKeyByInputArgs(out_grad);
  auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();

  Backend kernel_backend = kernel_key.backend();
  DataLayout kernel_layout = kernel_key.layout();
  DataType kernel_data_type = kernel_key.dtype();

  auto kernel = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "concat_grad", {kernel_backend, kernel_layout, kernel_data_type});
  VLOG(6) << "concat_grad API kernel key: [" << kernel_backend << ", "
          << kernel_layout << ", " << kernel_data_type << "]";
  VLOG(6) << "concat_grad API kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_backend);

  // std::unique_ptr<std::vector<phi::DenseTensor>>
  auto dense_x = PrepareData(x, kernel.InputAt(0), {});
  auto dense_out_grad = PrepareData(out_grad, kernel.InputAt(1), {});

  // Calculate the number of out tensors
  size_t out_number = x.size();
  std::vector<Tensor> x_grad;
  auto dense_x_grad = SetKernelOutput(out_number, kernel_backend, &x_grad);

  std::vector<phi::MetaTensor> meta_x;
  meta_x.reserve(x.size());
  std::vector<phi::MetaTensor*> meta_x_ptrs;
  meta_x_ptrs.reserve(x.size());
  for (const auto& t : *dense_x) {
    meta_x.push_back(t);
    meta_x_ptrs.push_back(&meta_x.back());
  }

  std::vector<phi::MetaTensor> meta_x_grad;
  meta_x_grad.reserve(x.size());
  std::vector<phi::MetaTensor*> meta_x_grad_ptrs;
  meta_x_grad_ptrs.reserve(x.size());
  for (size_t i = 0; i < out_number; ++i) {
    meta_x_grad.push_back(*dense_x_grad[i]);
    meta_x_grad_ptrs.push_back(&meta_x_grad.back());
  }

  phi::UnchangedMultiInferMeta(meta_x_ptrs, meta_x_grad_ptrs);

  std::vector<const phi::DenseTensor*> dense_x_ptr;
  dense_x_ptr.reserve(x.size());
  for (const auto& t : *dense_x) {
    dense_x_ptr.push_back(&t);
  }

  using kernel_signature = void (*)(const platform::DeviceContext&,
                                    const std::vector<const phi::DenseTensor*>&,
                                    const phi::DenseTensor&,
                                    const phi::Scalar&,
                                    std::vector<phi::DenseTensor*>);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  (*kernel_fn)(
      *dev_ctx, dense_x_ptr, *dense_out_grad, phi::Scalar(axis), dense_x_grad);

  return x_grad;
}

}  // namespace experimental
}  // namespace paddle
