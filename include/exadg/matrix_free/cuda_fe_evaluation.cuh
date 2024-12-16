/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2021 by the ExaDG authors
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *  ______________________________________________________________________
 */

#ifndef INCLUDE_EXADG_MATRIX_FREE_CUDA_FE_EVALUATION_CUH_
#define INCLUDE_EXADG_MATRIX_FREE_CUDA_FE_EVALUATION_CUH_

// deal.II
#include <deal.II/base/config.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/utilities.h>

// ExaDG
#include <exadg/matrix_free/cuda_matrix_free.h>
#include <exadg/matrix_free/cuda_tensor_product_kernels.cuh>

#include <cuda/std/array>

template<typename T, size_t N>
__host__ __device__ inline cuda::std::array<T, N>
         operator+(const cuda::std::array<T, N> & v1, const cuda::std::array<T, N> & v2)
{
  cuda::std::array<T, N> output{};
  for(auto i = 0U; i < N; ++i)
    output[i] = v1[i] + v2[i];
  return output;
}

template<typename T, size_t N>
__host__ __device__ inline cuda::std::array<T, N>
         operator-(const cuda::std::array<T, N> & v1, const cuda::std::array<T, N> & v2)
{
  cuda::std::array<T, N> output{};
  for(auto i = 0U; i < N; ++i)
    output[i] = v1[i] - v2[i];
  return output;
}

template<typename T, size_t N, typename V>
__host__ __device__ inline cuda::std::array<T, N>
         operator*(const cuda::std::array<T, N> & v1, const V scaler)
{
  cuda::std::array<T, N> output{};
  for(auto i = 0U; i < N; ++i)
    output[i] = v1[i] * scaler;
  return output;
}


namespace ExaDG
{
namespace CUDAWrappers
{
/**
 * Compute the dof/quad index for a given thread id, dimension, and
 * number of points in each space dimensions.
 */
template<int dim, int n_points_1d>
__device__ inline unsigned int
compute_index()
{
  return (dim == 1 ? threadIdx.x % n_points_1d :
                     threadIdx.x + n_points_1d * (threadIdx.y % n_points_1d));
}

/**
 * For face integral, compute the dof/quad index for a given thread id,
 * dimension, and number of points in each space dimensions.
 */
template<int dim, int n_points_1d>
__device__ inline unsigned int
compute_face_index(unsigned int face_number, unsigned int z = 0)
{
  face_number = face_number / 2;

  return (dim == 1 ? 0 :
          dim == 2 ? (face_number == 0 ? threadIdx.y % n_points_1d : threadIdx.x) :
                     (face_number == 0 ? threadIdx.y % n_points_1d + n_points_1d * z :
                      face_number == 1 ? threadIdx.x + n_points_1d * z :
                                         threadIdx.x + n_points_1d * (threadIdx.y % n_points_1d)));

  // FIX: The following is the correct one with distorted mesh,
  // but somehow there is a bug in permutation when running with TensorCores.
  // face_number == 1 ? threadIdx.x * n_points_1d + z :
}

/*----------------------- Helper functions ---------------------------------*/
/**
 * Compute the quadrature point index in the local cell of a given thread.
 */
template<int dim, int n_q_points_1d>
__device__ inline unsigned int
q_point_id_in_cell()
{
  return (dim == 1 ? threadIdx.x % n_q_points_1d :
                     threadIdx.x + n_q_points_1d * (threadIdx.y % n_q_points_1d));
}

/**
 * Return the quadrature point index local of a given thread. The index is
 * only unique for a given MPI process.
 */
template<int dim, typename Number>
__device__ inline unsigned int
local_q_point_id(const unsigned int                             cell,
                 const typename MatrixFree<dim, Number>::Data * data,
                 const unsigned int                             n_q_points_1d,
                 const unsigned int                             n_q_points)
{
  return (data->row_start / data->padding_length + cell) * n_q_points +
         q_point_id_in_cell<dim>(n_q_points_1d);
}

/**
 * Return the quadrature point associated with a given thread.
 */
template<int dim, typename Number, int n_q_points_1d>
__device__ inline dealii::Point<dim, Number> &
get_quadrature_point(const unsigned int                             cell,
                     const typename MatrixFree<dim, Number>::Data * data,
                     const unsigned int                             index)
{
  return *(data->q_points + data->padding_length * cell + index);
}


/**
 * This class provides all the functions necessary to evaluate functions at
 * quadrature points and cell integrations. In functionality, this class is
 * similar to FEValues<dim>.
 *
 * This class has five template arguments:
 *
 * @tparam dim Dimension in which this class is to be used
 *
 * @tparam fe_degree Degree of the tensor prodict finite element with fe_degree+1
 * degrees of freedom per coordinate direction
 *
 * @tparam n_q_points_1d Number of points in the quadrature formular in 1D,
 * defaults to fe_degree+1
 *
 * @tparam n_components Number of vector components when solving a system of
 * PDEs. If the same operation is applied to several components of a PDE (e.g.
 * a vector Laplace equation), they can be applied simultaneously with one
 * call (and often more efficiently). Defaults to 1
 *
 * @tparam Number Number format, @p double or @p float. Defaults to @p
 * double.
 *
 * @ingroup MatrixFree
 */
template<int dim,
         int fe_degree,
         int n_q_points_1d = fe_degree + 1,
         int n_components_ = 1,
         typename Number   = double>
class FEEvaluation
{
public:
  static constexpr unsigned int n_dofs_z = dim == 3 ? fe_degree + 1 : 1;

  /**
   * An alias for scalar quantities.
   */
  using value_type = cuda::std::array<Number, n_dofs_z>;

  /**
   * An alias for vectorial quantities.
   */
  using gradient_type = cuda::std::array<value_type, dim>;

  /**
   * An alias for vectorial quantities.
   */
  using hessian_type = cuda::std::array<gradient_type, dim>;

  /**
   * An alias to kernel specific information.
   */
  using data_type = typename MatrixFree<dim, Number>::Data;

  /**
   * Dimension.
   */
  static constexpr unsigned int dimension = dim;

  /**
   * Number of components.
   */
  static constexpr unsigned int n_components = n_components_;

  /**
   * Number of quadrature points per cell.
   */
  static constexpr unsigned int n_q_points = dealii::Utilities::pow(n_q_points_1d, dim);

  /**
   * Number of tensor degrees of freedoms per cell.
   */
  static constexpr unsigned int tensor_dofs_per_cell = dealii::Utilities::pow(fe_degree + 1, dim);

  /**
   * Constructor.
   */
  __device__
  FEEvaluation(const unsigned int        cell_id,
               const data_type *         data,
               SharedData<dim, Number> * shdata);

  /**
   * Constructor on host.
   */
  __host__
  FEEvaluation(const data_type * data, SharedData<dim, Number> * shdata);

  /**
   * Initialize the operation pointer to the current cell index.
   */
  __device__ void
  reinit(const unsigned int cell_id);

  /**
   * For the vector @p src, read out the values on the degrees of freedom of
   * the current cell, and store them internally. Similar functionality as
   * the function DoFAccessor::get_interpolated_dof_values when no
   * constraints are present, but it also includes constraints from hanging
   * nodes, so once can see it as a similar function to
   * AffineConstraints::read_dof_valuess as well.
   */
  __device__ void
  read_dof_values(const Number * src);

  /**
   * Take the value stored internally on dof values of the current cell and
   * sum them into the vector @p dst. The function also applies constraints
   * during the write operation. The functionality is hence similar to the
   * function AffineConstraints::distribute_local_to_global.
   */
  __device__ void
  distribute_local_to_global(Number * dst) const;

  /**
   * Evaluate the function values and the gradients of the FE function given
   * at the DoF values in the input vector at the quadrature points on the
   * unit cell. The function arguments specify which parts shall actually be
   * computed. This function needs to be called before the functions
   * @p get_value() or @p get_gradient() give useful information.
   */
  __device__ void
  evaluate(const bool evaluate_val, const bool evaluate_grad);

  /**
   * Evaluate the function hessians of the FE function given at the DoF values
   * in the input vector at the quadrature points on the unit cell. The
   * function arguments specify which parts shall actually be computed. This
   * function needs to be called before the functions @p get_hessian() give
   * useful information.
   * @warning only the diagonal elements
   * @todo full hessian matrix
   */
  __device__ void
  evaluate_hessian();

  /**
   * This function takes the values and/or gradients that are stored on
   * quadrature points, tests them by all the basis functions/gradients on
   * the cell and performs the cell integration. The two function arguments
   * @p integrate_val and @p integrate_grad are used to enable/disable some
   * of the values or the gradients.
   */
  __device__ void
  integrate(const bool integrate_val, const bool integrate_grad);

  /**
   * Same as above, except that the quadrature point is computed from thread
   * id.
   */
  __device__ value_type
  get_value() const;

  /**
   * Same as above, except that the local dof index is computed from the
   * thread id.
   */
  __device__ value_type
  get_dof_value() const;

  /**
   * Same as above, except that the quadrature point is computed from the
   * thread id.
   */
  __device__ void
  submit_value(const value_type & val_in);

  /**
   * Same as above, except that the local dof index is computed from the
   * thread id.
   */
  __device__ void
  submit_dof_value(const value_type & val_in);

  /**
   * Same as above, except that the quadrature point is computed from the
   * thread id.
   */
  __device__ gradient_type
  get_gradient() const;

  /**
   * Same as above, except that the quadrature point is computed from the
   * thread id.
   */
  __device__ value_type
  get_trace_hessian() const;

  /**
   * Same as above, except that the quadrature point is computed from the
   * thread id.
   */
  __device__ void
  submit_gradient(const gradient_type & grad_in);

  // clang-format off
    /**
     * Same as above, except that the functor @p func only takes a single input
     * argument (fe_eval) and computes the quadrature point from the thread id.
     *
     * @p func needs to define
     * \code
     * __device__ void operator()(
     *   CUDAWrappers::FEEvaluation<dim, fe_degree, n_q_points_1d, n_components, Number> *fe_eval) const;
     * \endcode
     */
  // clang-format on
  template<typename Functor>
  __device__ void
  apply_for_each_quad_point(const Functor & func);

private:
  unsigned int n_cells;
  unsigned int padding_length;

  // TODO: need optimization
  bool is_cartesian;


  dealii::types::global_dof_index * local_to_global;

  // Internal buffer
  Number * values;
  Number * gradients[dim];

  Number * JxW;
  Number * inv_jac;

  Number * shape_values;
  Number * shape_gradients;
  Number * co_shape_gradients;
};


template<int dim, int fe_degree, int n_q_points_1d, int n_components_, typename Number>
__device__
FEEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::FEEvaluation(
  const unsigned int        cell_id,
  const data_type *         data,
  SharedData<dim, Number> * shdata)
  : n_cells(data -> n_cells),
    padding_length(data->padding_length),
    is_cartesian(data->geometry_type ==
                 dealii::internal::MatrixFreeFunctions::GeometryType::cartesian),
    values(shdata->values),
    shape_values(shdata->shape_values),
    shape_gradients(shdata->shape_gradients),
    co_shape_gradients(shdata->co_shape_gradients)
{
  static_assert(n_components_ == 1, "This function only supports FE with one \
                  components");

  if(is_cartesian)
  {
    inv_jac = data->inv_jacobian;
    JxW     = data->JxW;
  }
  else
  {
    inv_jac = data->inv_jacobian + padding_length * cell_id;
    JxW     = data->JxW + padding_length * cell_id;
  }

  local_to_global = data->local_to_global + cell_id;

  for(unsigned int i = 0; i < dim; ++i)
    gradients[i] = shdata->gradients[i];

  // TODO: permutation with TensorCores
  const unsigned int idx = compute_index<dim, n_q_points_1d>();

  shape_values[idx]       = data->cell_face_shape_values[idx];
  shape_gradients[idx]    = data->cell_face_shape_gradients[idx];
  co_shape_gradients[idx] = data->cell_face_co_shape_gradients[idx];
}

template<int dim, int fe_degree, int n_q_points_1d, int n_components_, typename Number>
__host__
FEEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::FEEvaluation(
  const data_type *         data,
  SharedData<dim, Number> * shdata)
  : n_cells(data -> n_cells),
    padding_length(data->padding_length),
    is_cartesian(data->geometry_type ==
                 dealii::internal::MatrixFreeFunctions::GeometryType::cartesian),
    values(shdata->values),
    shape_values(shdata->shape_values),
    shape_gradients(shdata->shape_gradients),
    co_shape_gradients(shdata->co_shape_gradients)
{
  static_assert(n_components_ == 1, "This function only supports FE with one \
                  components");

  inv_jac = data->inv_jacobian;
  JxW     = data->JxW;

  local_to_global = data->local_to_global;

  for(unsigned int i = 0; i < dim; ++i)
    gradients[i] = shdata->gradients[i];

  // TODO: permutation with TensorCores
}

template<int dim, int fe_degree, int n_q_points_1d, int n_components_, typename Number>
__device__ void
FEEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::reinit(
  const unsigned int cell_id)
{
  if(!is_cartesian)
  {
    inv_jac = inv_jac + padding_length * cell_id;
    JxW     = JxW + padding_length * cell_id;
  }

  local_to_global = local_to_global + cell_id;
}

template<int dim, int fe_degree, int n_q_points_1d, int n_components_, typename Number>
__device__ void
FEEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::read_dof_values(
  const Number * src)
{
  const dealii::types::global_dof_index src_idx = local_to_global[0];

  for(unsigned int i = 0; i < n_dofs_z; ++i)
  {
    const unsigned int idx =
      compute_index<dim, n_q_points_1d>() + i * n_q_points_1d * n_q_points_1d;

    // Use the read-only data cache.
    values[idx] = __ldg(&src[src_idx + idx]);
  }

  __syncthreads();
}

template<int dim, int fe_degree, int n_q_points_1d, int n_components_, typename Number>
__device__ void
FEEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::distribute_local_to_global(
  Number * dst) const
{
  const dealii::types::global_dof_index destination_idx = local_to_global[0];

  for(unsigned int i = 0; i < n_dofs_z; ++i)
  {
    const unsigned int idx =
      compute_index<dim, n_q_points_1d>() + i * n_q_points_1d * n_q_points_1d;

    atomicAdd(&dst[destination_idx + idx], values[idx]);
  }
}

template<int dim, int fe_degree, int n_q_points_1d, int n_components_, typename Number>
__device__ void
FEEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::evaluate(
  const bool evaluate_val,
  const bool evaluate_grad)
{
  // First evaluate the gradients because it requires values that will be
  // changed if evaluate_val is true
  EvaluatorTensorProduct<EvaluatorVariant::evaluate_general, dim, fe_degree, n_q_points_1d, Number>
    evaluator_tensor_product(shape_values, shape_gradients, co_shape_gradients);

  if(evaluate_val == true && evaluate_grad == true)
  {
    evaluator_tensor_product.value_and_gradient_at_quad_pts(values, gradients);
    __syncthreads();
  }
  else if(evaluate_grad == true)
  {
    evaluator_tensor_product.gradient_at_quad_pts(values, gradients);
    __syncthreads();
  }
  else if(evaluate_val == true)
  {
    evaluator_tensor_product.value_at_quad_pts(values);
    __syncthreads();
  }
}

template<int dim, int fe_degree, int n_q_points_1d, int n_components_, typename Number>
__device__ void
FEEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::integrate(
  const bool integrate_val,
  const bool integrate_grad)
{
  EvaluatorTensorProduct<EvaluatorVariant::evaluate_general, dim, fe_degree, n_q_points_1d, Number>
    evaluator_tensor_product(shape_values, shape_gradients, co_shape_gradients);

  if(integrate_val == true && integrate_grad == true)
  {
    evaluator_tensor_product.integrate_value_and_gradient(values, gradients);
    __syncthreads();
  }
  else if(integrate_val == true)
  {
    evaluator_tensor_product.integrate_value(values);
    __syncthreads();
  }
  else if(integrate_grad == true)
  {
    evaluator_tensor_product.integrate_gradient<false>(values, gradients);
    __syncthreads();
  }
}

template<int dim, int fe_degree, int n_q_points_1d, int n_components_, typename Number>
__device__ typename FEEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::value_type
FEEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::get_value() const
{
  value_type val;

  for(unsigned int i = 0; i < n_dofs_z; ++i)
  {
    const unsigned int q_point =
      i * n_q_points_1d * n_q_points_1d + compute_index<dim, n_q_points_1d>();
    val[i] = values[q_point];
  }
  return val;
}

template<int dim, int fe_degree, int n_q_points_1d, int n_components_, typename Number>
__device__ typename FEEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::value_type
FEEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::get_dof_value() const
{
  value_type val;

  for(unsigned int i = 0; i < n_dofs_z; ++i)
  {
    const unsigned int dof =
      i * n_q_points_1d * n_q_points_1d + compute_index<dim, n_q_points_1d>();
    val[i] = values[dof];
  }
  return val;
}

template<int dim, int fe_degree, int n_q_points_1d, int n_components_, typename Number>
__device__ void
FEEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::submit_value(
  const value_type & val_in)
{
  for(unsigned int i = 0; i < n_dofs_z; ++i)
  {
    const unsigned int q_point =
      i * n_q_points_1d * n_q_points_1d + compute_index<dim, n_q_points_1d>();

    values[q_point] = val_in[i] * JxW[q_point];
  }

  __syncthreads();
}

template<int dim, int fe_degree, int n_q_points_1d, int n_components_, typename Number>
__device__ void
FEEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::submit_dof_value(
  const value_type & val_in)
{
  for(unsigned int i = 0; i < n_dofs_z; ++i)
  {
    const unsigned int dof =
      i * n_q_points_1d * n_q_points_1d + compute_index<dim, fe_degree + 1>();
    values[dof] = val_in[i];
  }
  __syncthreads();
}

template<int dim, int fe_degree, int n_q_points_1d, int n_components_, typename Number>
__device__
  typename FEEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::gradient_type
  FEEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::get_gradient() const
{
  gradient_type grad;

  Number * inv_jacobian = &inv_jac[0];
  for(unsigned int i = 0; i < n_dofs_z; ++i)
  {
    const unsigned int q_point =
      i * n_q_points_1d * n_q_points_1d + compute_index<dim, n_q_points_1d>();

    if(!is_cartesian)
      inv_jacobian = &inv_jac[q_point];

    for(unsigned int d_1 = 0; d_1 < dim; ++d_1)
    {
      Number tmp = 0.;
      for(unsigned int d_2 = 0; d_2 < dim; ++d_2)
        tmp += inv_jacobian[padding_length * n_cells * (dim * d_2 + d_1)] * gradients[d_2][q_point];
      grad[d_1][i] = tmp;
    }
  }

  return grad;
}

template<int dim, int fe_degree, int n_q_points_1d, int n_components_, typename Number>
__device__ void
FEEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::submit_gradient(
  const gradient_type & grad_in)
{
  Number * inv_jacobian = &inv_jac[0];
  for(unsigned int i = 0; i < n_dofs_z; ++i)
  {
    const unsigned int q_point =
      (i * n_q_points_1d * n_q_points_1d + compute_index<dim, n_q_points_1d>());

    if(!is_cartesian)
      inv_jacobian = &inv_jac[q_point];

    for(unsigned int d_1 = 0; d_1 < dim; ++d_1)
    {
      Number tmp = 0.;
      for(unsigned int d_2 = 0; d_2 < dim; ++d_2)
        tmp += inv_jacobian[n_cells * padding_length * (dim * d_1 + d_2)] * grad_in[d_2][i];
      gradients[d_1][q_point] = tmp * JxW[q_point];
    }
  }

  __syncthreads();
}

template<int dim, int fe_degree, int n_q_points_1d, int n_components_, typename Number>
template<typename Functor>
__device__ void
FEEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::apply_for_each_quad_point(
  const Functor & func)
{
  func(this);

  __syncthreads();
}


/**
 * This class provides all the functions necessary to evaluate functions at
 * quadrature points and cell/face integrations. In functionality, this class
 * is similar to FEFaceValues<dim>.
 *
 * This class has five template arguments:
 *
 * @tparam dim Dimension in which this class is to be used
 *
 * @tparam fe_degree Degree of the tensor prodict finite element with fe_degree+1
 * degrees of freedom per coordinate direction
 *
 * @tparam n_q_points_1d Number of points in the quadrature formular in 1D,
 * defaults to fe_degree+1
 *
 * @tparam n_components Number of vector components when solving a system of
 * PDEs. If the same operation is applied to several components of a PDE (e.g.
 * a vector Laplace equation), they can be applied simultaneously with one
 * call (and often more efficiently). Defaults to 1
 *
 * @tparam Number Number format, @p double or @p float. Defaults to @p
 * double.
 *
 * @ingroup MatrixFree
 */
template<int dim,
         int fe_degree,
         int n_q_points_1d = fe_degree + 1,
         int n_components_ = 1,
         typename Number   = double>
class FEFaceEvaluation
{
public:
  static constexpr unsigned int n_dofs_z = dim == 3 ? fe_degree + 1 : 1;

  /**
   * An alias for scalar quantities.
   */
  using value_type = cuda::std::array<Number, n_dofs_z>;

  /**
   * An alias for vectorial quantities.
   */
  using gradient_type = cuda::std::array<value_type, dim>;

  /**
   * An alias to kernel specific information.
   */
  using data_type = typename MatrixFree<dim, Number>::Data;

  /**
   * Dimension.
   */
  static constexpr unsigned int dimension = dim;

  /**
   * Number of components.
   */
  static constexpr unsigned int n_components = n_components_;

  /**
   * Number of quadrature points per cell.
   */
  static constexpr unsigned int n_q_points = dealii::Utilities::pow(n_q_points_1d, dim - 1);

  /**
   * Number of tensor degrees of freedoms per cell.
   */
  static constexpr unsigned int tensor_dofs_per_cell = dealii::Utilities::pow(fe_degree + 1, dim);

  /**
   * Constructor.
   */
  __device__
  FEFaceEvaluation(const unsigned int        face_id,
                   const data_type *         data,
                   SharedData<dim, Number> * shdata,
                   const bool                is_interior_face = true);

  /**
   * Constructor on host.
   */
  __host__
  FEFaceEvaluation(const data_type * data, SharedData<dim, Number> * shdata);

  /**
   * Initialize the operation pointer to the current cell index.
   */
  __device__ void
  reinit(const unsigned int face_id, const bool is_interior_face = true);

  /**
   * For the vector @p src, read out the values on the degrees of freedom of
   * the current cell, and store them internally. Similar functionality as
   * the function DoFAccessor::get_interpolated_dof_values when no
   * constraints are present, but it also includes constraints from hanging
   * nodes, so once can see it as a similar function to
   * AffineConstraints::read_dof_valuess as well.
   */
  __device__ void
  read_dof_values(const Number * src);

  /**
   * Take the value stored internally on dof values of the current cell and
   * sum them into the vector @p dst. The function also applies constraints
   * during the write operation. The functionality is hence similar to the
   * function AffineConstraints::distribute_local_to_global.
   */
  __device__ void
  distribute_local_to_global(Number * dst) const;

  /**
   * Evaluates the function values, the gradients, and the Laplacians of the
   * FE function given at the DoF values stored in the internal data field
   * dof_values (that is usually filled by the read_dof_values() method) at
   * the quadrature points on the unit cell. The function arguments specify
   * which parts shall actually be computed. Needs to be called before the
   * functions get_value(), get_gradient() or get_normal_derivative() give
   * useful information (unless these values have been set manually by
   * accessing the internal data pointers).
   */
  __device__ void
  evaluate(const bool evaluate_val, const bool evaluate_grad);

  /**
   * This function takes the values and/or gradients that are stored on
   * quadrature points, tests them by all the basis functions/gradients on the
   * cell and performs the cell integration. The two function arguments
   * integrate_val and integrate_grad are used to enable/disable some of
   * values or gradients. The result is written into the internal data field
   * dof_values (that is usually written into the result vector by the
   * distribute_local_to_global() or set_dof_values() methods).
   */
  __device__ void
  integrate(const bool integrate_val, const bool integrate_grad);

  /**
   * Same as above, except that the quadrature point is computed from thread
   * id.
   */
  __device__ value_type
  get_value() const;

  /**
   * Same as above, except that the local dof index is computed from the
   * thread id.
   */
  __device__ value_type
  get_dof_value() const;

  /**
   * Same as above, except that the quadrature point is computed from the
   * thread id.
   */
  __device__ void
  submit_value(const value_type & val_in);

  /**
   * Same as above, except that the local dof index is computed from the
   * thread id.
   */
  __device__ void
  submit_dof_value(const value_type & val_in);

  /**
   * Same as above, except that the quadrature point is computed from the
   * thread id.
   */
  __device__ gradient_type
  get_gradient() const;

  /**
   * Same as above, except that the quadrature point is computed from the
   * thread id.
   */
  __device__ void
  submit_gradient(const gradient_type & grad_in);

  /**
   * Same as above, except that the quadrature point is computed from the
   * thread id.
   */
  __device__ value_type
  get_normal_derivative() const;

  /**
   * Same as above, except that the quadrature point is computed from the
   * thread id.
   */
  __device__ void
  submit_normal_derivative(const value_type & grad_in);

  /**
   * length h_i normal to the face. For a general non-Cartesian mesh, this
   * length must be computed by the product of the inverse Jacobian times the
   * normal vector in real coordinates.
   */
  __device__ Number
  inverse_length_normal_to_face();

  // clang-format off
    /**
     * Same as above, except that the functor @p func only takes a single input
     * argument (fe_eval) and computes the quadrature point from the thread id.
     *
     * @p func needs to define
     * \code
     * __device__ void operator()(
     *   CUDAWrappers::FEEvaluation<dim, fe_degree, n_q_points_1d, n_components, Number> *fe_eval) const;
     * \endcode
     */
  // clang-format on
  template<typename Functor>
  __device__ void
  apply_for_each_quad_point(const Functor & func);


private:
  unsigned int face_id;
  unsigned int n_faces;
  unsigned int face_padding_length;
  unsigned int face_number;


  bool is_interior_face;
  bool is_cartesian;

  dealii::types::global_dof_index * local_to_global;

  // Internal buffer
  Number * values;
  Number * gradients[dim];

  Number * JxW;
  Number * inv_jac;
  Number * normal_vec;

  Number * shape_values;
  Number * shape_gradients;
  Number * co_shape_gradients;
};

template<int dim, int fe_degree, int n_q_points_1d, int n_components_, typename Number>
__device__
FEFaceEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::FEFaceEvaluation(
  const unsigned int        face_id,
  const data_type *         data,
  SharedData<dim, Number> * shdata,
  const bool                is_interior_face)
  : n_faces(data -> n_faces),
    face_padding_length(data->face_padding_length),
    is_interior_face(is_interior_face),
    is_cartesian(data->geometry_type ==
                 dealii::internal::MatrixFreeFunctions::GeometryType::cartesian),
    shape_values(shdata->shape_values),
    shape_gradients(shdata->shape_gradients),
    co_shape_gradients(shdata->co_shape_gradients)
{
  static_assert(n_components_ == 1, "This function only supports FE with one \
                  components");

  auto face_no = face_id;

  local_to_global = data->face_local_to_global + face_no;

  if(is_cartesian)
  {
    inv_jac = data->face_inv_jacobian;
    JxW     = data->face_JxW;
  }
  else
  {
    inv_jac = data->face_inv_jacobian + face_padding_length * face_no;
    JxW     = data->face_JxW + face_padding_length * face_no;
  }

  normal_vec  = data->normal_vector + face_padding_length * face_no;
  face_number = data->face_number[face_no];

  unsigned int shift = is_interior_face ? 0 : tensor_dofs_per_cell;

  values = &shdata->values[shift];

  for(unsigned int i = 0; i < dim; ++i)
    gradients[i] = &shdata->gradients[i][shift];

  // TODO: permutation with TensorCores
  const unsigned int index = compute_index<dim, n_q_points_1d>();

  for(unsigned int i = 0; i < 3; ++i)
  {
    auto idx = i * n_q_points_1d * n_q_points_1d + index;

    shape_values[idx]       = data->cell_face_shape_values[idx];
    shape_gradients[idx]    = data->cell_face_shape_gradients[idx];
    co_shape_gradients[idx] = data->cell_face_co_shape_gradients[idx];
  }
}

template<int dim, int fe_degree, int n_q_points_1d, int n_components_, typename Number>
__host__
FEFaceEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::FEFaceEvaluation(
  const data_type *         data,
  SharedData<dim, Number> * shdata)
  : n_faces(data -> n_faces),
    face_padding_length(data->face_padding_length),
    is_interior_face(is_interior_face),
    is_cartesian(data->geometry_type ==
                 dealii::internal::MatrixFreeFunctions::GeometryType::cartesian),
    shape_values(shdata->shape_values),
    shape_gradients(shdata->shape_gradients),
    co_shape_gradients(shdata->co_shape_gradients)
{
  static_assert(n_components_ == 1, "This function only supports FE with one \
                  components");

  inv_jac = data->face_inv_jacobian;
  JxW     = data->face_JxW;

  local_to_global = data->local_to_global;

  normal_vec  = data->normal_vector;
  face_number = data->face_number;

  unsigned int shift = is_interior_face ? 0 : tensor_dofs_per_cell;

  values = &shdata->values[shift];

  for(unsigned int i = 0; i < dim; ++i)
    gradients[i] = &shdata->gradients[i][shift];

  // TODO: permutation with TensorCores
}

template<int dim, int fe_degree, int n_q_points_1d, int n_components_, typename Number>
__device__ void
FEFaceEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::reinit(
  const unsigned int face_id,
  const bool         is_interior_face)
{
  auto face_no = is_interior_face ? face_id : face_id + 1;

  local_to_global = local_to_global + face_no;

  if(!is_cartesian)
  {
    inv_jac = inv_jac + face_padding_length * face_no;
    JxW     = JxW + face_padding_length * face_no;
  }

  normal_vec  = normal_vec + face_padding_length * face_no;
  face_number = face_number + face_no;
}

template<int dim, int fe_degree, int n_q_points_1d, int n_components_, typename Number>
__device__ void
FEFaceEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::read_dof_values(
  const Number * src)
{
  const dealii::types::global_dof_index src_idx = local_to_global[0];

  for(unsigned int i = 0; i < n_dofs_z; ++i)
  {
    const unsigned int idx =
      compute_index<dim, n_q_points_1d>() + i * n_q_points_1d * n_q_points_1d;

    // Use the read-only data cache.
    values[idx] = __ldg(&src[src_idx + idx]);
  }

  __syncthreads();
}

template<int dim, int fe_degree, int n_q_points_1d, int n_components_, typename Number>
__device__ void
FEFaceEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::distribute_local_to_global(
  Number * dst) const
{
  const dealii::types::global_dof_index destination_idx = local_to_global[0];

  for(unsigned int i = 0; i < n_dofs_z; ++i)
  {
    const unsigned int idx =
      compute_index<dim, n_q_points_1d>() + i * n_q_points_1d * n_q_points_1d;

    atomicAdd(&dst[destination_idx + idx], values[idx]);
  }
}

template<int dim, int fe_degree, int n_q_points_1d, int n_components_, typename Number>
__device__ void
FEFaceEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::evaluate(
  const bool evaluate_val,
  const bool evaluate_grad)
{
  // First evaluate the gradients because it requires values that will be
  // changed if evaluate_val is true
  EvaluatorTensorProduct<EvaluatorVariant::evaluate_face, dim, fe_degree, n_q_points_1d, Number>
    evaluator_tensor_product(face_number, shape_values, shape_gradients, co_shape_gradients);

  if(evaluate_val == true && evaluate_grad == true)
  {
    // TODO:
    // evaluator_tensor_product.value_and_gradient_at_quad_pts(values,
    //                                                         gradients);

    evaluator_tensor_product.gradient_at_quad_pts(values, gradients);
    __syncthreads();

    evaluator_tensor_product.value_at_quad_pts(values);
    __syncthreads();
  }
  else if(evaluate_grad == true)
  {
    evaluator_tensor_product.gradient_at_quad_pts(values, gradients);
    __syncthreads();
  }
  else if(evaluate_val == true)
  {
    evaluator_tensor_product.value_at_quad_pts(values);
    __syncthreads();
  }
}

template<int dim, int fe_degree, int n_q_points_1d, int n_components_, typename Number>
__device__ void
FEFaceEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::integrate(
  const bool integrate_val,
  const bool integrate_grad)
{
  // First evaluate the gradients because it requires values that will be
  // changed if evaluate_val is true
  EvaluatorTensorProduct<EvaluatorVariant::evaluate_face, dim, fe_degree, n_q_points_1d, Number>
    evaluator_tensor_product(face_number, shape_values, shape_gradients, co_shape_gradients);

  if(integrate_val == true && integrate_grad == true)
  {
    // TODO:
    // evaluator_tensor_product.integrate_value_and_gradient(values,
    //                                                       gradients);

    evaluator_tensor_product.integrate_value(values);
    __syncthreads();

    evaluator_tensor_product.integrate_gradient<true>(values, gradients);
    __syncthreads();
  }
  else if(integrate_val == true)
  {
    evaluator_tensor_product.integrate_value(values);
    __syncthreads();
  }
  else if(integrate_grad == true)
  {
    evaluator_tensor_product.integrate_gradient<false>(values, gradients);
    __syncthreads();
  }
}

template<int dim, int fe_degree, int n_q_points_1d, int n_components_, typename Number>
__device__
  typename FEFaceEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::value_type
  FEFaceEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::get_value() const
{
  value_type val;

  for(unsigned int i = 0; i < n_dofs_z; ++i)
  {
    const unsigned int q_point =
      i * n_q_points_1d * n_q_points_1d + compute_index<dim, n_q_points_1d>();
    val[i] = values[q_point];
  }
  return val;
}


template<int dim, int fe_degree, int n_q_points_1d, int n_components_, typename Number>
__device__
  typename FEFaceEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::value_type
  FEFaceEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::get_dof_value() const
{
  value_type val;

  for(unsigned int i = 0; i < n_dofs_z; ++i)
  {
    const unsigned int dof =
      i * n_q_points_1d * n_q_points_1d + compute_index<dim, fe_degree + 1>();
    val[i] = values[dof];
  }
  return val;
}

template<int dim, int fe_degree, int n_q_points_1d, int n_components_, typename Number>
__device__ void
FEFaceEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::submit_value(
  const value_type & val_in)
{
  for(unsigned int i = 0; i < n_dofs_z; ++i)
  {
    const unsigned int q_point =
      i * n_q_points_1d * n_q_points_1d + compute_index<dim, n_q_points_1d>();
    const unsigned int q_point_face = compute_face_index<dim, n_q_points_1d>(face_number, i);

    values[q_point] = val_in[i] * JxW[q_point_face];
  }

  __syncthreads();
}



template<int dim, int fe_degree, int n_q_points_1d, int n_components_, typename Number>
__device__ void
FEFaceEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::submit_dof_value(
  const value_type & val_in)
{
  for(unsigned int i = 0; i < n_dofs_z; ++i)
  {
    const unsigned int dof =
      i * n_q_points_1d * n_q_points_1d + compute_index<dim, fe_degree + 1>();
    values[dof] = val_in[i];
  }

  __syncthreads();
}

template<int dim, int fe_degree, int n_q_points_1d, int n_components_, typename Number>
__device__
  typename FEFaceEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::gradient_type
  FEFaceEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::get_gradient() const
{
  Number * inv_jacobian = &inv_jac[0];

  gradient_type grad;

  for(unsigned int i = 0; i < n_dofs_z; ++i)
  {
    const unsigned int q_point =
      i * n_q_points_1d * n_q_points_1d + compute_index<dim, n_q_points_1d>();

    if(!is_cartesian)
    {
      auto q_point_face = compute_face_index<dim, n_q_points_1d>(face_number, i);
      inv_jacobian      = &inv_jac[q_point_face];
    }

    for(unsigned int d_1 = 0; d_1 < dim; ++d_1)
    {
      Number tmp = 0.;
      for(unsigned int d_2 = 0; d_2 < dim; ++d_2)
        tmp +=
          inv_jacobian[n_faces * face_padding_length * (dim * d_2 + d_1)] * gradients[d_2][q_point];
      grad[d_1][i] = tmp;
    }
  }

  return grad;
}

template<int dim, int fe_degree, int n_q_points_1d, int n_components_, typename Number>
__device__ void
FEFaceEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::submit_gradient(
  const gradient_type & grad_in)
{
  Number * inv_jacobian = &inv_jac[0];

  for(unsigned int i = 0; i < n_dofs_z; ++i)
  {
    const unsigned int q_point =
      i * n_q_points_1d * n_q_points_1d + compute_index<dim, n_q_points_1d>();
    const unsigned int q_point_face = compute_face_index<dim, n_q_points_1d>(face_number, i);

    if(!is_cartesian)
      inv_jacobian = &inv_jac[q_point_face];

    for(unsigned int d_1 = 0; d_1 < dim; ++d_1)
    {
      Number tmp = 0.;
      for(unsigned int d_2 = 0; d_2 < dim; ++d_2)
        tmp += inv_jacobian[n_faces * face_padding_length * (dim * d_1 + d_2)] * grad_in[d_2][i];
      gradients[d_1][q_point] = tmp * JxW[q_point_face];
    }
  }

  __syncthreads();
}

template<int dim, int fe_degree, int n_q_points_1d, int n_components_, typename Number>
__device__
  typename FEFaceEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::value_type
  FEFaceEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::get_normal_derivative()
    const
{
  Number * normal_vector = &normal_vec[0];

  const Number coe = is_interior_face ? 1.0 : -1.0;

  gradient_type grad              = get_gradient();
  value_type    normal_derivative = {};

  for(unsigned int i = 0; i < n_dofs_z; ++i)
  {
    if(!is_cartesian)
    {
      auto q_point_face = compute_face_index<dim, n_q_points_1d>(face_number, i);
      normal_vector     = &normal_vec[q_point_face];
    }

    for(unsigned int d = 0; d < dim; ++d)
      normal_derivative[i] += grad[d][i] * normal_vector[n_faces * face_padding_length * d] * coe;
  }

  return normal_derivative;
}

template<int dim, int fe_degree, int n_q_points_1d, int n_components_, typename Number>
__device__ void
FEFaceEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::submit_normal_derivative(
  const value_type & grad_in)
{
  Number * normal_vector = &normal_vec[0];
  Number * inv_jacobian  = &inv_jac[0];

  const Number coe = is_interior_face ? 1. : -1.;

  gradient_type normal_x_jacobian;

  for(unsigned int i = 0; i < n_dofs_z; ++i)
  {
    const unsigned int q_point =
      i * n_q_points_1d * n_q_points_1d + compute_index<dim, n_q_points_1d>();
    const unsigned int q_point_face = compute_face_index<dim, n_q_points_1d>(face_number, i);

    if(!is_cartesian)
    {
      normal_vector = &normal_vec[q_point_face];
      inv_jacobian  = &inv_jac[q_point_face];
    }

    for(unsigned int d_1 = 0; d_1 < dim; ++d_1)
    {
      Number tmp = 0.;
      for(unsigned int d_2 = 0; d_2 < dim; ++d_2)
        tmp += inv_jacobian[n_faces * face_padding_length * (dim * d_1 + d_2)] *
               normal_vector[n_faces * face_padding_length * d_2];
      normal_x_jacobian[d_1][i] = coe * tmp;
    }

    for(unsigned int d = 0; d < dim; ++d)
      gradients[d][q_point] = grad_in[i] * normal_x_jacobian[d][i] * JxW[q_point_face];
  }

  __syncthreads();
}

template<int dim, int fe_degree, int n_q_points_1d, int n_components_, typename Number>
__device__ Number
FEFaceEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::
  inverse_length_normal_to_face()
{
  Number tmp = 0.;
  for(unsigned int d = 0; d < dim; ++d)
    tmp += inv_jac[n_faces * face_padding_length * (dim * (face_number / 2) + d)] *
           normal_vec[n_faces * face_padding_length * d];

  return tmp;
}

template<int dim, int fe_degree, int n_q_points_1d, int n_components_, typename Number>
template<typename Functor>
__device__ void
FEFaceEvaluation<dim, fe_degree, n_q_points_1d, n_components_, Number>::apply_for_each_quad_point(
  const Functor & func)
{
  func(this);

  __syncthreads();
}

} // namespace CUDAWrappers
} // namespace ExaDG

#endif /* INCLUDE_EXADG_MATRIX_FREE_CUDA_FE_EVALUATION_CUH_ */
