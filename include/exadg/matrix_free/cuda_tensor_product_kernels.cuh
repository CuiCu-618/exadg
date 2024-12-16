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

#ifndef INCLUDE_EXADG_MATRIX_FREE_CUDA_TENSOR_PRODUCT_KERNEL_CUH_
#define INCLUDE_EXADG_MATRIX_FREE_CUDA_TENSOR_PRODUCT_KERNEL_CUH_

// deal.II
#include <deal.II/base/config.h>
#include <deal.II/base/utilities.h>

#include <mma.h>
#include <type_traits>

using namespace nvcuda;

namespace ExaDG
{
namespace CUDAWrappers
{

// Function to calculate GCD of two numbers
__device__ constexpr unsigned int
gcd(unsigned int a, unsigned int b)
{
  if(b == 0)
    return a;
  return gcd(b, a % b);
}

// Recursive template function to calculate LCM of two numbers
template<int a, int b>
struct LCM
{
  static constexpr unsigned int value = (a * b) / gcd(a, b);
};

// Function to calculate the multiple of a number
template<int n, int constant>
__device__ constexpr unsigned int
calculate_multiple()
{
  // Calculate the multiple of n
  constexpr unsigned int multiple = LCM<n, constant>::value / n;

  return multiple;
}

template<int n_dofs_1d, typename Number = double>
__host__ __device__ inline unsigned int
get_base(const unsigned int row, const unsigned int z = 0)
{
  printf("Should never be called!\n");
  return 0;
}

template<>
__host__ __device__ inline unsigned int
get_base<8, double>(const unsigned int row, const unsigned int z)
{
  auto base1 = (row & 3) < 2 ? 0 : 4;
  auto base2 = (z & 1) << 3;
  auto base3 = (z & 3) < 2 ? 0 : 4;

  return base1 ^ base2 ^ base3;
}

template<int n_dofs_1d, typename Number = double>
__host__ __device__ inline unsigned int
get_face_base(const unsigned int face_number, const unsigned int row, const unsigned int z = 0)
{
  printf("Should never be called!\n");
  return 0;
}

template<>
__host__ __device__ inline unsigned int
get_face_base<8, double>(const unsigned int face_number,
                         const unsigned int row,
                         const unsigned int z)
{
  if(face_number == 0)
    return (z & 1);
  else if(face_number == 1)
    return ((row & 3) < 2 ? 0 : 4) ^ ((z & 3) < 2 ? 0 : 4);
  else
    return get_base<8, double>(row, z);
}


/**
 * In this namespace, the evaluator routines that evaluate the tensor
 * products are implemented.
 *
 * @ingroup MatrixFree
 */
// TODO: for now only the general variant and face are implemented
enum EvaluatorVariant
{
  /**
   * Do not use anything more than the tensor product structure of the finite
   * element.
   */
  evaluate_general,
  /**
   * Perform evaluation by exploiting symmetry in the finite element: i.e.,
   * skip some computations by utilizing the symmetry in the shape functions
   * and quadrature points.
   */
  evaluate_symmetric,
  /**
   * Raviart-Thomas elements with anisotropic polynomials.
   */
  evaluate_raviart_thomas,
  /**
   * Tensor product structure on faces.
   */
  evaluate_face
};

/**
 * Generic evaluator framework.
 *
 * @ingroup MatrixFree
 */
template<EvaluatorVariant variant, int dim, int fe_degree, int n_q_points_1d, typename Number>
struct EvaluatorTensorProduct;

/**
 * Internal evaluator for 1d-3d shape function using the tensor product form
 * of the basis functions.
 *
 * @ingroup MatrixFree
 */
template<int dim, int fe_degree, int n_q_points_1d, typename Number>
struct EvaluatorTensorProduct<evaluate_general, dim, fe_degree, n_q_points_1d, Number>
{
  static constexpr unsigned int dofs_per_cell = dealii::Utilities::pow(fe_degree + 1, dim);
  static constexpr unsigned int n_q_points    = dealii::Utilities::pow(n_q_points_1d, dim);

  static constexpr unsigned int n_dofs_z = dim == 3 ? fe_degree + 1 : 1;

  __device__
  EvaluatorTensorProduct(Number * shv, Number * shg, Number * co_shg);

  /**
   * Evaluate the values of a finite element function at the quadrature
   * points.
   */
  template<int direction, bool dof_to_quad, bool add, bool in_place>
  __device__ void
  values(Number shape_values[], const Number * in, Number * out) const;

  /**
   * Evaluate the gradient of a finite element function at the quadrature
   * points for a given @p direction.
   */
  template<int direction, bool dof_to_quad, bool add, bool in_place>
  __device__ void
  gradients(Number shape_gradients[], const Number * in, Number * out) const;

  /**
   * Helper function for values() and gradients().
   */
  template<int direction, bool dof_to_quad, bool add, bool in_place>
  __device__ void
  apply(Number shape_data[], const Number * in, Number * out) const;

  /**
   * Evaluate the finite element function at the quadrature points.
   */
  __device__ void
  value_at_quad_pts(Number * u);

  /**
   * Helper function for integrate(). Integrate the finite element function.
   */
  __device__ void
  integrate_value(Number * u);

  /**
   * Evaluate the gradients of the finite element function at the quadrature
   * points.
   */
  __device__ void
  gradient_at_quad_pts(const Number * const u, Number * grad_u[dim]);

  /**
   * Evaluate the diagnoal of hessian of the finite element function at the
   * quadrature points.
   */
  __device__ void
  hessian_at_quad_pts(const Number * const u, Number * grad_u[dim]);

  /**
   * Evaluate the values and the gradients of the finite element function at
   * the quadrature points.
   */
  __device__ void
  value_and_gradient_at_quad_pts(Number * const u, Number * grad_u[dim]);

  /**
   * Helper function for integrate(). Integrate the gradients of the finite
   * element function.
   */
  template<bool add>
  __device__ void
  integrate_gradient(Number * u, Number * grad_u[dim]);

  /**
   * Helper function for integrate(). Integrate the values and the gradients
   * of the finite element function.
   */
  __device__ void
  integrate_value_and_gradient(Number * u, Number * grad_u[dim]);

  Number * shape_values;
  Number * shape_gradients;
  Number * co_shape_gradients;
};


template<int dim, int fe_degree, int n_q_points_1d, typename Number>
__device__
EvaluatorTensorProduct<evaluate_general, dim, fe_degree, n_q_points_1d, Number>::
  EvaluatorTensorProduct(Number * shv, Number * shg, Number * co_shg)
  : shape_values(shv), shape_gradients(shg), co_shape_gradients(co_shg)
{
}

template<int dim, int fe_degree, int n_q_points_1d, typename Number>
template<int direction, bool dof_to_quad, bool add, bool in_place>
__device__ void
EvaluatorTensorProduct<evaluate_general, dim, fe_degree, n_q_points_1d, Number>::values(
  Number         shape_values[],
  const Number * in,
  Number *       out) const
{
  apply<direction, dof_to_quad, add, in_place>(shape_values, in, out);
}

template<int dim, int fe_degree, int n_q_points_1d, typename Number>
template<int direction, bool dof_to_quad, bool add, bool in_place>
__device__ void
EvaluatorTensorProduct<evaluate_general, dim, fe_degree, n_q_points_1d, Number>::gradients(
  Number         shape_gradients[],
  const Number * in,
  Number *       out) const
{
  apply<direction, dof_to_quad, add, in_place>(shape_gradients, in, out);
}

/*
template<int dim, int fe_degree, int n_q_points_1d, typename Number>
template<int direction, bool dof_to_quad, bool add, bool in_place>
__device__ typename std::enable_if<fe_degree == 7 && std::is_same<Number, double>::value>::type
EvaluatorTensorProduct<evaluate_general, dim, fe_degree, n_q_points_1d, Number>::apply(
  Number         shape_data[],
  const Number * in,
  Number *       out) const
{
  if(direction == 0)
  {
    const int tid    = (threadIdx.y * 8 + threadIdx.x) & 31;
    const int warpId = threadIdx.y / 4;

    const int row = tid / 4;
    const int col = tid & 3;

    constexpr int offset = n_q_points_1d * n_q_points_1d;

    double2 c[n_q_points_1d / 2];
    for(int z = 0; z < n_q_points_1d / 2; ++z)
    {
      const int c_idx = (row * n_q_points_1d + 2 * col + (z * 2 + warpId) * offset) ^
                        get_base<n_q_points_1d>(row, z * 2 + warpId);

      if constexpr(add)
        c[z] = *((double2 *)(out + c_idx));
      else
        c[z] = {0, 0};
    }

    if(add)
      __syncthreads();

    for(int cycle = 0; cycle < 2; ++cycle)
    {
      const int b_idx =
        dof_to_quad ?
          ((col + cycle * 4) * n_q_points_1d + row) ^ get_base<n_q_points_1d>(col + cycle * 4, 0) :
          (col + cycle * 4 + n_q_points_1d * row) ^ get_base<n_q_points_1d>(row, 0);

      auto b0 = shape_data[b_idx];

      for(int z = 0; z < n_q_points_1d / 2; ++z)
      {
        const int a_idx = (row * n_q_points_1d + col + cycle * 4 + (z * 2 + warpId) * offset) ^
                          get_base<n_q_points_1d>(row, z * 2 + warpId);

        auto a0 = in_place ? out[a_idx] : in[a_idx];

        asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
                     "{%0,%1}, {%2}, {%3}, {%4,%5};\n"
                     : "=d"(c[z].x), "=d"(c[z].y)
                     : "d"(a0), "d"(b0), "d"(c[z].x), "d"(c[z].y));
      }
    }

    if(in_place)
      __syncthreads();

    for(int z = 0; z < n_q_points_1d / 2; ++z)
    {
      const int c_idx = (row * n_q_points_1d + 2 * col + (z * 2 + warpId) * offset) ^
                        get_base<n_q_points_1d>(row, z * 2 + warpId);

      *((double2 *)(out + c_idx)) = c[z];
    }
  }
  else if(direction == 1)
  {
    const int tid    = (threadIdx.y * 8 + threadIdx.x) & 31;
    const int warpId = threadIdx.y / 4;

    const int row = tid / 4;
    const int col = tid & 3;

    constexpr int offset = n_q_points_1d * n_q_points_1d;

    double2 c[n_q_points_1d / 2];
    for(int z = 0; z < n_q_points_1d / 2; ++z)
    {
      const int c_idx = (row * n_q_points_1d + 2 * col + (z * 2 + warpId) * offset) ^
                        get_base<n_q_points_1d>(row, z * 2 + warpId);

      if constexpr(add)
        c[z] = *((double2 *)(out + c_idx));
      else
        c[z] = {0, 0};
    }

    if(add)
      __syncthreads();

    for(int cycle = 0; cycle < 2; ++cycle)
    {
      const int a_idx =
        dof_to_quad ?
          (row + n_q_points_1d * (col + cycle * 4)) ^ get_base<n_q_points_1d>(col + cycle * 4, 0) :
          (row * n_q_points_1d + col + cycle * 4) ^ get_base<n_q_points_1d>(row, 0);
      auto a0 = shape_data[a_idx];

      for(int z = 0; z < n_q_points_1d / 2; ++z)
      {
        const int b_idx = ((col + cycle * 4) * n_q_points_1d + row + (z * 2 + warpId) * offset) ^
                          get_base<n_q_points_1d>(col + cycle * 4, z * 2 + warpId);

        auto b0 = in_place ? out[b_idx] : in[b_idx];

        asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
                     "{%0,%1}, {%2}, {%3}, {%4,%5};\n"
                     : "=d"(c[z].x), "=d"(c[z].y)
                     : "d"(a0), "d"(b0), "d"(c[z].x), "d"(c[z].y));
      }
    }

    if(in_place)
      __syncthreads();

    for(int z = 0; z < n_q_points_1d / 2; ++z)
    {
      const int c_idx = (row * n_q_points_1d + 2 * col + (z * 2 + warpId) * offset) ^
                        get_base<n_q_points_1d>(row, z * 2 + warpId);

      *((double2 *)(out + c_idx)) = c[z];
    }
  }
  else
  {
    const int tid    = (threadIdx.y * 8 + threadIdx.x) & 31;
    const int warpId = threadIdx.y / 4;

    const int row = tid / 4;
    const int col = tid & 3;

    constexpr int offset = n_q_points_1d * n_q_points_1d;

    double2 c[n_q_points_1d / 2];
    for(int z = 0; z < n_q_points_1d / 2; ++z)
    {
      const int c_idx = ((z * 2 + warpId) * n_q_points_1d + 2 * col + row * offset) ^
                        get_base<n_q_points_1d>(z * 2 + warpId, row);

      if constexpr(add)
        c[z] = *((double2 *)(out + c_idx));
      else
        c[z] = {0, 0};
    }

    if(add)
      __syncthreads();

    for(int cycle = 0; cycle < 2; ++cycle)
    {
      const int a_idx =
        dof_to_quad ?
          (row + n_q_points_1d * (col + cycle * 4)) ^ get_base<n_q_points_1d>(col + cycle * 4, 0) :
          (row * n_q_points_1d + col + cycle * 4) ^ get_base<n_q_points_1d>(row, 0);
      auto a0 = shape_data[a_idx];

      for(int z = 0; z < n_q_points_1d / 2; ++z)
      {
        const int b_idx = ((z * 2 + warpId) * n_q_points_1d + row + (col + cycle * 4) * offset) ^
                          get_base<n_q_points_1d>(z * 2 + warpId, col + cycle * 4);

        auto b0 = in_place ? out[b_idx] : in[b_idx];

        asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
                     "{%0,%1}, {%2}, {%3}, {%4,%5};\n"
                     : "=d"(c[z].x), "=d"(c[z].y)
                     : "d"(a0), "d"(b0), "d"(c[z].x), "d"(c[z].y));
      }
    }

    if(in_place)
      __syncthreads();

    for(int z = 0; z < n_q_points_1d / 2; ++z)
    {
      const int c_idx = ((z * 2 + warpId) * n_q_points_1d + 2 * col + row * offset) ^
                        get_base<n_q_points_1d>(z * 2 + warpId, row);

      *((double2 *)(out + c_idx)) = c[z];
    }
  }
}
*/


template<int dim, int fe_degree, int n_q_points_1d, typename Number>
template<int direction, bool dof_to_quad, bool add, bool in_place>
__device__ void
EvaluatorTensorProduct<evaluate_general, dim, fe_degree, n_q_points_1d, Number>::apply(
  Number         shape_data[],
  const Number * in,
  Number *       out) const
{
  constexpr unsigned int multiple = calculate_multiple<n_q_points_1d, 16>();

  const unsigned int row = threadIdx.y % n_q_points_1d;
  const unsigned int col = threadIdx.x;

  Number t[n_dofs_z];

  for(unsigned int z = 0; z < n_dofs_z; ++z)
  {
    t[z] = 0;
    for(unsigned int k = 0; k < n_q_points_1d; ++k)
    {
      const unsigned int shape_idx =
        dof_to_quad ?
          ((direction == 0) ? (col + ((k + col / multiple) % n_q_points_1d) * n_q_points_1d) :
           (direction == 1) ? (row + k * n_q_points_1d) :
                              (z + k * n_q_points_1d)) :
          ((direction == 0) ? ((k + col / multiple) % n_q_points_1d + col * n_q_points_1d) :
           (direction == 1) ? (k + row * n_q_points_1d) :
                              (k + z * n_q_points_1d));
      const unsigned int source_idx =
        (direction == 0) ?
          ((k + col / multiple) % n_q_points_1d + n_q_points_1d * (row + n_q_points_1d * z)) :
        (direction == 1) ? (col + n_q_points_1d * (k + n_q_points_1d * z)) :
                           (col + n_q_points_1d * (row + n_q_points_1d * k));
      t[z] += shape_data[shape_idx] * (in_place ? out[source_idx] : in[source_idx]);
    }
  }

  if(in_place)
    __syncthreads();

  for(unsigned int z = 0; z < n_dofs_z; ++z)
  {
    const unsigned int destination_idx = col + n_q_points_1d * (row + n_q_points_1d * z);

    if(add)
      out[destination_idx] += t[z];
    else
      out[destination_idx] = t[z];
  }
}



template<int dim, int fe_degree, int n_q_points_1d, typename Number>
inline __device__ void
EvaluatorTensorProduct<evaluate_general, dim, fe_degree, n_q_points_1d, Number>::value_at_quad_pts(
  Number * u)
{
  switch(dim)
  {
    case 1:
    {
      values<0, true, false, true>(shape_values, u, u);

      break;
    }
    case 2:
    {
      values<0, true, false, true>(shape_values, u, u);
      __syncthreads();
      values<1, true, false, true>(shape_values, u, u);

      break;
    }
    case 3:
    {
      values<0, true, false, true>(shape_values, u, u);
      __syncthreads();
      values<1, true, false, true>(shape_values, u, u);
      __syncthreads();
      values<2, true, false, true>(shape_values, u, u);

      break;
    }
    default:
    {
    }
  }
}



template<int dim, int fe_degree, int n_q_points_1d, typename Number>
inline __device__ void
EvaluatorTensorProduct<evaluate_general, dim, fe_degree, n_q_points_1d, Number>::integrate_value(
  Number * u)
{
  switch(dim)
  {
    case 1:
    {
      values<0, false, false, true>(shape_values, u, u);

      break;
    }
    case 2:
    {
      values<0, false, false, true>(shape_values, u, u);
      __syncthreads();

      values<1, false, false, true>(shape_values, u, u);

      break;
    }
    case 3:
    {
      values<0, false, false, true>(shape_values, u, u);
      __syncthreads();
      values<1, false, false, true>(shape_values, u, u);
      __syncthreads();
      values<2, false, false, true>(shape_values, u, u);

      break;
    }
    default:
    {
    }
  }
}



template<int dim, int fe_degree, int n_q_points_1d, typename Number>
inline __device__ void
EvaluatorTensorProduct<evaluate_general, dim, fe_degree, n_q_points_1d, Number>::
  gradient_at_quad_pts(const Number * const u, Number * grad_u[dim])
{
  switch(dim)
  {
    case 1:
    {
      gradients<0, true, false, false>(shape_values, u, grad_u[0]);

      break;
    }
    case 2:
    {
      gradients<0, true, false, false>(shape_gradients, u, grad_u[0]);
      values<0, true, false, false>(shape_values, u, grad_u[1]);

      __syncthreads();

      values<1, true, false, true>(shape_values, grad_u[0], grad_u[0]);
      gradients<1, true, false, true>(shape_gradients, grad_u[1], grad_u[1]);

      break;
    }
    case 3:
    {
      gradients<0, true, false, false>(shape_gradients, u, grad_u[0]);
      values<0, true, false, false>(shape_values, u, grad_u[1]);
      values<0, true, false, false>(shape_values, u, grad_u[2]);

      __syncthreads();

      values<1, true, false, true>(shape_values, grad_u[0], grad_u[0]);
      gradients<1, true, false, true>(shape_gradients, grad_u[1], grad_u[1]);
      values<1, true, false, true>(shape_values, grad_u[2], grad_u[2]);

      __syncthreads();

      values<2, true, false, true>(shape_values, grad_u[0], grad_u[0]);
      values<2, true, false, true>(shape_values, grad_u[1], grad_u[1]);
      gradients<2, true, false, true>(shape_gradients, grad_u[2], grad_u[2]);

      break;
    }
    default:
    {
    }
  }
}

template<int dim, int fe_degree, int n_q_points_1d, typename Number>
inline __device__ void
EvaluatorTensorProduct<evaluate_general, dim, fe_degree, n_q_points_1d, Number>::
  value_and_gradient_at_quad_pts(Number * const u, Number * grad_u[dim])
{
  switch(dim)
  {
    case 1:
    {
      values<0, true, false, true>(shape_values, u, u);
      __syncthreads();

      gradients<0, true, false, false>(co_shape_gradients, u, grad_u[0]);

      break;
    }
    case 2:
    {
      values<0, true, false, true>(shape_values, u, u);
      __syncthreads();
      values<1, true, false, true>(shape_values, u, u);
      __syncthreads();

      gradients<0, true, false, false>(co_shape_gradients, u, grad_u[0]);
      gradients<1, true, false, false>(co_shape_gradients, u, grad_u[1]);

      break;
    }
    case 3:
    {
      values<0, true, false, true>(shape_values, u, u);
      __syncthreads();
      values<1, true, false, true>(shape_values, u, u);
      __syncthreads();
      values<2, true, false, true>(shape_values, u, u);
      __syncthreads();

      gradients<0, true, false, false>(co_shape_gradients, u, grad_u[0]);
      gradients<1, true, false, false>(co_shape_gradients, u, grad_u[1]);
      gradients<2, true, false, false>(co_shape_gradients, u, grad_u[2]);

      break;
    }
    default:
    {
    }
  }
}



template<int dim, int fe_degree, int n_q_points_1d, typename Number>
template<bool add>
inline __device__ void
EvaluatorTensorProduct<evaluate_general, dim, fe_degree, n_q_points_1d, Number>::integrate_gradient(
  Number * u,
  Number * grad_u[dim])
{
  switch(dim)
  {
    case 1:
    {
      gradients<0, false, add, false>(shape_gradients, grad_u[dim], u);

      break;
    }
    case 2:
    {
      gradients<0, false, false, true>(shape_gradients, grad_u[0], grad_u[0]);
      values<0, false, false, true>(shape_values, grad_u[1], grad_u[1]);

      __syncthreads();

      values<1, false, add, false>(shape_values, grad_u[0], u);
      __syncthreads();
      gradients<1, false, true, false>(shape_gradients, grad_u[1], u);

      break;
    }
    case 3:
    {
      gradients<0, false, false, true>(shape_gradients, grad_u[0], grad_u[0]);
      values<0, false, false, true>(shape_values, grad_u[1], grad_u[1]);
      values<0, false, false, true>(shape_values, grad_u[2], grad_u[2]);

      __syncthreads();

      values<1, false, false, true>(shape_values, grad_u[0], grad_u[0]);
      gradients<1, false, false, true>(shape_gradients, grad_u[1], grad_u[1]);
      values<1, false, false, true>(shape_values, grad_u[2], grad_u[2]);

      __syncthreads();

      values<2, false, add, false>(shape_values, grad_u[0], u);
      __syncthreads();
      values<2, false, true, false>(shape_values, grad_u[1], u);
      __syncthreads();
      gradients<2, false, true, false>(shape_gradients, grad_u[2], u);

      break;
    }
    default:
    {
    }
  }
}



template<int dim, int fe_degree, int n_q_points_1d, typename Number>
inline __device__ void
EvaluatorTensorProduct<evaluate_general, dim, fe_degree, n_q_points_1d, Number>::
  integrate_value_and_gradient(Number * u, Number * grad_u[dim])
{
  switch(dim)
  {
    case 1:
    {
      gradients<0, false, true, false>(co_shape_gradients, grad_u[0], u);
      __syncthreads();

      values<0, false, false, true>(shape_values, u, u);

      break;
    }
    case 2:
    {
      gradients<1, false, true, false>(co_shape_gradients, grad_u[1], u);
      __syncthreads();
      gradients<0, false, true, false>(co_shape_gradients, grad_u[0], u);
      __syncthreads();

      values<1, false, false, true>(shape_values, u, u);
      __syncthreads();
      values<0, false, false, true>(shape_values, u, u);
      __syncthreads();

      break;
    }
    case 3:
    {
      gradients<2, false, true, false>(co_shape_gradients, grad_u[2], u);
      __syncthreads();
      gradients<1, false, true, false>(co_shape_gradients, grad_u[1], u);
      __syncthreads();
      gradients<0, false, true, false>(co_shape_gradients, grad_u[0], u);
      __syncthreads();

      values<2, false, false, true>(shape_values, u, u);
      __syncthreads();
      values<1, false, false, true>(shape_values, u, u);
      __syncthreads();
      values<0, false, false, true>(shape_values, u, u);
      __syncthreads();

      break;
    }
    default:
    {
    }
  }
}


/**
 * Internal evaluator for 1d-3d shape function using the tensor product form
 * of the basis functions, including face integral.
 *
 * @ingroup MatrixFree
 */
template<int dim, int fe_degree, int n_q_points_1d, typename Number>
struct EvaluatorTensorProduct<evaluate_face, dim, fe_degree, n_q_points_1d, Number>
{
  static constexpr unsigned int dofs_per_cell = dealii::Utilities::pow(fe_degree + 1, dim);

  static constexpr unsigned int n_dofs_z = dim == 3 ? fe_degree + 1 : 1;

  __device__
  EvaluatorTensorProduct(unsigned int face_number, Number * shv, Number * shg, Number * co_shg);

  /**
   * Evaluate the values of a finite element function at the quadrature
   * points.
   */
  template<int direction, bool dof_to_quad, bool add, bool in_place>
  __device__ void
  values(Number shape_values[], const Number * in, Number * out) const;

  /**
   * Evaluate the gradient of a finite element function at the quadrature
   * points for a given @p direction.
   */
  template<int direction, bool dof_to_quad, bool add, bool in_place>
  __device__ void
  gradients(Number shape_gradients[], const Number * in, Number * out) const;

  /**
   * Helper function for values() and gradients().
   */
  template<int direction, bool dof_to_quad, bool add, bool in_place>
  __device__ void
  apply(Number shape_data[], const Number * in, Number * out) const;

  /**
   * Evaluate the finite element function at the quadrature points.
   */
  __device__ void
  value_at_quad_pts(Number * u);

  /**
   * Helper function for integrate(). Integrate the finite element function.
   */
  __device__ void
  integrate_value(Number * u);

  /**
   * Evaluate the gradients of the finite element function at the quadrature
   * points.
   */
  __device__ void
  gradient_at_quad_pts(const Number * const u, Number * grad_u[dim]);

  /**
   * Evaluate the values and the gradients of the finite element function at
   * the quadrature points.
   */
  __device__ void
  value_and_gradient_at_quad_pts(Number * const u, Number * grad_u[dim]);

  /**
   * Helper function for integrate(). Integrate the gradients of the finite
   * element function.
   */
  template<bool add>
  __device__ void
  integrate_gradient(Number * u, Number * grad_u[dim]);

  /**
   * Helper function for integrate(). Integrate the values and the gradients
   * of the finite element function.
   */
  __device__ void
  integrate_value_and_gradient(Number * u, Number * grad_u[dim]);

  const unsigned int face_number;

  Number * shape_values;
  Number * shape_gradients;
  Number * co_shape_gradients;
};


template<int dim, int fe_degree, int n_q_points_1d, typename Number>
__device__
EvaluatorTensorProduct<evaluate_face, dim, fe_degree, n_q_points_1d, Number>::
  EvaluatorTensorProduct(unsigned int face_number, Number * shv, Number * shg, Number * co_shg)
  : face_number(face_number), shape_values(shv), shape_gradients(shg), co_shape_gradients(co_shg)
{
}

template<int dim, int fe_degree, int n_q_points_1d, typename Number>
template<int direction, bool dof_to_quad, bool add, bool in_place>
__device__ void
EvaluatorTensorProduct<evaluate_face, dim, fe_degree, n_q_points_1d, Number>::values(
  Number         shape_values[],
  const Number * in,
  Number *       out) const
{
  apply<direction, dof_to_quad, add, in_place>(shape_values, in, out);
}



template<int dim, int fe_degree, int n_q_points_1d, typename Number>
template<int direction, bool dof_to_quad, bool add, bool in_place>
__device__ void
EvaluatorTensorProduct<evaluate_face, dim, fe_degree, n_q_points_1d, Number>::gradients(
  Number         shape_gradients[],
  const Number * in,
  Number *       out) const
{
  apply<direction, dof_to_quad, add, in_place>(shape_gradients, in, out);
}

/*
template<int dim, int fe_degree, int n_q_points_1d, typename Number>
template<int direction, bool dof_to_quad, bool add, bool in_place>
__device__ typename std::enable_if<fe_degree == 7 && std::is_same<Number, double>::value>::type
EvaluatorTensorProduct<evaluate_face, dim, fe_degree, n_q_points_1d, Number>::apply(
  Number         shape_data[],
  const Number * in,
  Number *       out) const
{
  if(direction == 0)
  {
    const int tid    = (threadIdx.y * 8 + threadIdx.x) & 31;
    const int warpId = threadIdx.y / 4;

    const int row = tid / 4;
    const int col = tid & 3;

    constexpr int offset = n_q_points_1d * n_q_points_1d;

    double2 c[n_q_points_1d / 2];
    for(int z = 0; z < n_q_points_1d / 2; ++z)
    {
      const int c_idx = (row * n_q_points_1d + 2 * col + (z * 2 + warpId) * offset) ^
                        get_base<n_q_points_1d>(row, z * 2 + warpId);

      if constexpr(add)
        c[z] = *((double2 *)(out + c_idx));
      else
        c[z] = {0, 0};
    }

    if(add)
      __syncthreads();

    for(int cycle = 0; cycle < 2; ++cycle)
    {
      const int b_idx =
        dof_to_quad ?
          ((col + cycle * 4) * n_q_points_1d + row) ^ get_base<n_q_points_1d>(col + cycle * 4, 0) :
          (col + cycle * 4 + n_q_points_1d * row) ^ get_base<n_q_points_1d>(row, 0);

      auto b0 = shape_data[b_idx];

      for(int z = 0; z < n_q_points_1d / 2; ++z)
      {
        const int a_idx = (row * n_q_points_1d + col + cycle * 4 + (z * 2 + warpId) * offset) ^
                          get_base<n_q_points_1d>(row, z * 2 + warpId);

        auto a0 = in_place ? out[a_idx] : in[a_idx];

        asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
                     "{%0,%1}, {%2}, {%3}, {%4,%5};\n"
                     : "=d"(c[z].x), "=d"(c[z].y)
                     : "d"(a0), "d"(b0), "d"(c[z].x), "d"(c[z].y));
      }
    }

    if(in_place)
      __syncthreads();

    for(int z = 0; z < n_q_points_1d / 2; ++z)
    {
      const int c_idx = (row * n_q_points_1d + 2 * col + (z * 2 + warpId) * offset) ^
                        get_base<n_q_points_1d>(row, z * 2 + warpId);

      *((double2 *)(out + c_idx)) = c[z];
    }
  }
  else if(direction == 1)
  {
    const int tid    = (threadIdx.y * 8 + threadIdx.x) & 31;
    const int warpId = threadIdx.y / 4;

    const int row = tid / 4;
    const int col = tid & 3;

    constexpr int offset = n_q_points_1d * n_q_points_1d;

    double2 c[n_q_points_1d / 2];
    for(int z = 0; z < n_q_points_1d / 2; ++z)
    {
      const int c_idx = (row * n_q_points_1d + 2 * col + (z * 2 + warpId) * offset) ^
                        get_base<n_q_points_1d>(row, z * 2 + warpId);

      if constexpr(add)
        c[z] = *((double2 *)(out + c_idx));
      else
        c[z] = {0, 0};
    }

    if(add)
      __syncthreads();

    for(int cycle = 0; cycle < 2; ++cycle)
    {
      const int a_idx =
        dof_to_quad ?
          (row + n_q_points_1d * (col + cycle * 4)) ^ get_base<n_q_points_1d>(col + cycle * 4, 0) :
          (row * n_q_points_1d + col + cycle * 4) ^ get_base<n_q_points_1d>(row, 0);
      auto a0 = shape_data[a_idx];

      for(int z = 0; z < n_q_points_1d / 2; ++z)
      {
        const int b_idx = ((col + cycle * 4) * n_q_points_1d + row + (z * 2 + warpId) * offset) ^
                          get_base<n_q_points_1d>(col + cycle * 4, z * 2 + warpId);

        auto b0 = in_place ? out[b_idx] : in[b_idx];

        asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
                     "{%0,%1}, {%2}, {%3}, {%4,%5};\n"
                     : "=d"(c[z].x), "=d"(c[z].y)
                     : "d"(a0), "d"(b0), "d"(c[z].x), "d"(c[z].y));
      }
    }

    if(in_place)
      __syncthreads();

    for(int z = 0; z < n_q_points_1d / 2; ++z)
    {
      const int c_idx = (row * n_q_points_1d + 2 * col + (z * 2 + warpId) * offset) ^
                        get_base<n_q_points_1d>(row, z * 2 + warpId);

      *((double2 *)(out + c_idx)) = c[z];
    }
  }
  else
  {
    const int tid    = (threadIdx.y * 8 + threadIdx.x) & 31;
    const int warpId = threadIdx.y / 4;

    const int row = tid / 4;
    const int col = tid & 3;

    constexpr int offset = n_q_points_1d * n_q_points_1d;

    double2 c[n_q_points_1d / 2];
    for(int z = 0; z < n_q_points_1d / 2; ++z)
    {
      const int c_idx = ((z * 2 + warpId) * n_q_points_1d + 2 * col + row * offset) ^
                        get_base<n_q_points_1d>(z * 2 + warpId, row);

      if constexpr(add)
        c[z] = *((double2 *)(out + c_idx));
      else
        c[z] = {0, 0};
    }

    if(add)
      __syncthreads();

    for(int cycle = 0; cycle < 2; ++cycle)
    {
      const int a_idx =
        dof_to_quad ?
          (row + n_q_points_1d * (col + cycle * 4)) ^ get_base<n_q_points_1d>(col + cycle * 4, 0) :
          (row * n_q_points_1d + col + cycle * 4) ^ get_base<n_q_points_1d>(row, 0);
      auto a0 = shape_data[a_idx];

      for(int z = 0; z < n_q_points_1d / 2; ++z)
      {
        const int b_idx = ((z * 2 + warpId) * n_q_points_1d + row + (col + cycle * 4) * offset) ^
                          get_base<n_q_points_1d>(z * 2 + warpId, col + cycle * 4);

        auto b0 = in_place ? out[b_idx] : in[b_idx];

        asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
                     "{%0,%1}, {%2}, {%3}, {%4,%5};\n"
                     : "=d"(c[z].x), "=d"(c[z].y)
                     : "d"(a0), "d"(b0), "d"(c[z].x), "d"(c[z].y));
      }
    }

    if(in_place)
      __syncthreads();

    for(int z = 0; z < n_q_points_1d / 2; ++z)
    {
      const int c_idx = ((z * 2 + warpId) * n_q_points_1d + 2 * col + row * offset) ^
                        get_base<n_q_points_1d>(z * 2 + warpId, row);

      *((double2 *)(out + c_idx)) = c[z];
    }
  }
}
*/

template<int dim, int fe_degree, int n_q_points_1d, typename Number>
template<int direction, bool dof_to_quad, bool add, bool in_place>
__device__ void
EvaluatorTensorProduct<evaluate_face, dim, fe_degree, n_q_points_1d, Number>::apply(
  Number         shape_data[],
  const Number * in,
  Number *       out) const
{
  constexpr unsigned int multiple = calculate_multiple<n_q_points_1d, 16>();

  const unsigned int row = threadIdx.y % n_q_points_1d;
  const unsigned int col = threadIdx.x;

  Number t[n_dofs_z];

  for(unsigned int z = 0; z < n_dofs_z; ++z)
  {
    t[z] = 0;
    for(unsigned int k = 0; k < n_q_points_1d; ++k)
    {
      const unsigned int shape_idx =
        dof_to_quad ?
          ((direction == 0) ? (col + ((k + col / multiple) % n_q_points_1d) * n_q_points_1d) :
           (direction == 1) ? (row + k * n_q_points_1d) :
                              (z + k * n_q_points_1d)) :
          ((direction == 0) ? ((k + col / multiple) % n_q_points_1d + col * n_q_points_1d) :
           (direction == 1) ? (k + row * n_q_points_1d) :
                              (k + z * n_q_points_1d));
      const unsigned int source_idx =
        (direction == 0) ?
          ((k + col / multiple) % n_q_points_1d + n_q_points_1d * (row + n_q_points_1d * z)) :
        (direction == 1) ? (col + n_q_points_1d * (k + n_q_points_1d * z)) :
                           (col + n_q_points_1d * (row + n_q_points_1d * k));
      t[z] += shape_data[shape_idx] * (in_place ? out[source_idx] : in[source_idx]);
    }
  }

  if(in_place)
    __syncthreads();

  for(unsigned int z = 0; z < n_dofs_z; ++z)
  {
    const unsigned int destination_idx = col + n_q_points_1d * (row + n_q_points_1d * z);

    if(add)
      out[destination_idx] += t[z];
    else
      out[destination_idx] = t[z];
  }
}


template<int dim, int fe_degree, int n_q_points_1d, typename Number>
inline __device__ void
EvaluatorTensorProduct<evaluate_face, dim, fe_degree, n_q_points_1d, Number>::value_at_quad_pts(
  Number * u)
{
  constexpr unsigned int n_q_points_2d = n_q_points_1d * n_q_points_1d;

  const unsigned int shift = (face_number & 1) * n_q_points_2d;

  Number * shape_value_dir0 =
    face_number / 2 == 0 ? shape_values + n_q_points_2d + shift : shape_values;

  Number * shape_value_dir1 =
    face_number / 2 == 1 ? shape_values + n_q_points_2d + shift : shape_values;

  Number * shape_value_dir2 =
    face_number / 2 == 2 ? shape_values + n_q_points_2d + shift : shape_values;

  switch(dim)
  {
    case 1:
    {
      values<0, true, false, true>(shape_value_dir0, u, u);

      break;
    }
    case 2:
    {
      values<0, true, false, true>(shape_value_dir0, u, u);
      __syncthreads();
      values<1, true, false, true>(shape_value_dir1, u, u);

      break;
    }
    case 3:
    {
      values<0, true, false, true>(shape_value_dir0, u, u);
      __syncthreads();
      values<1, true, false, true>(shape_value_dir1, u, u);
      __syncthreads();
      values<2, true, false, true>(shape_value_dir2, u, u);

      break;
    }
    default:
    {
    }
  }
}



template<int dim, int fe_degree, int n_q_points_1d, typename Number>
inline __device__ void
EvaluatorTensorProduct<evaluate_face, dim, fe_degree, n_q_points_1d, Number>::integrate_value(
  Number * u)
{
  constexpr unsigned int n_q_points_2d = n_q_points_1d * n_q_points_1d;

  const unsigned int shift = (face_number & 1) * n_q_points_2d;

  Number * shape_value_dir0 =
    face_number / 2 == 0 ? shape_values + n_q_points_2d + shift : shape_values;

  Number * shape_value_dir1 =
    face_number / 2 == 1 ? shape_values + n_q_points_2d + shift : shape_values;

  Number * shape_value_dir2 =
    face_number / 2 == 2 ? shape_values + n_q_points_2d + shift : shape_values;

  switch(dim)
  {
    case 1:
    {
      values<0, false, false, true>(shape_value_dir0, u, u);

      break;
    }
    case 2:
    {
      values<0, false, false, true>(shape_value_dir0, u, u);
      __syncthreads();
      values<1, false, false, true>(shape_value_dir1, u, u);

      break;
    }
    case 3:
    {
      values<0, false, false, true>(shape_value_dir0, u, u);
      __syncthreads();
      values<1, false, false, true>(shape_value_dir1, u, u);
      __syncthreads();
      values<2, false, false, true>(shape_value_dir2, u, u);

      break;
    }
    default:
    {
    }
  }
}



template<int dim, int fe_degree, int n_q_points_1d, typename Number>
inline __device__ void
EvaluatorTensorProduct<evaluate_face, dim, fe_degree, n_q_points_1d, Number>::gradient_at_quad_pts(
  const Number * const u,
  Number *             grad_u[dim])
{
  constexpr unsigned int n_q_points_2d = n_q_points_1d * n_q_points_1d;

  const unsigned int shift = (face_number & 1) * n_q_points_2d;

  Number * shape_value_dir0 =
    face_number / 2 == 0 ? shape_values + n_q_points_2d + shift : shape_values;

  Number * shape_value_dir1 =
    face_number / 2 == 1 ? shape_values + n_q_points_2d + shift : shape_values;

  Number * shape_value_dir2 =
    face_number / 2 == 2 ? shape_values + n_q_points_2d + shift : shape_values;

  Number * shape_gradient_dir0 =
    face_number / 2 == 0 ? shape_gradients + n_q_points_2d + shift : shape_gradients;

  Number * shape_gradient_dir1 =
    face_number / 2 == 1 ? shape_gradients + n_q_points_2d + shift : shape_gradients;

  Number * shape_gradient_dir2 =
    face_number / 2 == 2 ? shape_gradients + n_q_points_2d + shift : shape_gradients;

  switch(dim)
  {
    case 1:
    {
      gradients<0, true, false, false>(shape_gradient_dir0, u, grad_u[0]);

      break;
    }
    case 2:
    {
      gradients<0, true, false, false>(shape_gradient_dir0, u, grad_u[0]);
      values<0, true, false, false>(shape_value_dir0, u, grad_u[1]);

      __syncthreads();

      values<1, true, false, true>(shape_value_dir1, grad_u[0], grad_u[0]);
      gradients<1, true, false, true>(shape_gradient_dir1, grad_u[1], grad_u[1]);

      break;
    }
    case 3:
    {
      gradients<0, true, false, false>(shape_gradient_dir0, u, grad_u[0]);
      values<0, true, false, false>(shape_value_dir0, u, grad_u[1]);
      values<0, true, false, false>(shape_value_dir0, u, grad_u[2]);

      __syncthreads();

      values<1, true, false, true>(shape_value_dir1, grad_u[0], grad_u[0]);
      gradients<1, true, false, true>(shape_gradient_dir1, grad_u[1], grad_u[1]);
      values<1, true, false, true>(shape_value_dir1, grad_u[2], grad_u[2]);

      __syncthreads();

      values<2, true, false, true>(shape_value_dir2, grad_u[0], grad_u[0]);
      values<2, true, false, true>(shape_value_dir2, grad_u[1], grad_u[1]);
      gradients<2, true, false, true>(shape_gradient_dir2, grad_u[2], grad_u[2]);

      break;
    }
    default:
    {
    }
  }
}



template<int dim, int fe_degree, int n_q_points_1d, typename Number>
inline __device__ void
EvaluatorTensorProduct<evaluate_face, dim, fe_degree, n_q_points_1d, Number>::
  value_and_gradient_at_quad_pts(Number * const u, Number * grad_u[dim])
{
  constexpr unsigned int n_q_points_2d = n_q_points_1d * n_q_points_1d;

  const unsigned int shift = (face_number & 1) * n_q_points_2d;

  Number * shape_value_dir0 =
    face_number / 2 == 0 ? shape_values + n_q_points_2d + shift : shape_values;

  Number * shape_value_dir1 =
    face_number / 2 == 1 ? shape_values + n_q_points_2d + shift : shape_values;

  Number * shape_value_dir2 =
    face_number / 2 == 2 ? shape_values + n_q_points_2d + shift : shape_values;

  // TODO:
  Number * co_shape_gradient_dir0 =
    face_number / 2 == 0 ? co_shape_gradients + shift : co_shape_gradients;

  Number * co_shape_gradient_dir1 =
    face_number / 2 == 1 ? co_shape_gradients + shift : co_shape_gradients;

  Number * co_shape_gradient_dir2 =
    face_number / 2 == 2 ? co_shape_gradients + shift : co_shape_gradients;

  switch(dim)
  {
    case 1:
    {
      values<0, true, false, true>(shape_value_dir0, u, u);
      __syncthreads();

      gradients<0, true, false, false>(co_shape_gradient_dir0, u, grad_u[0]);

      break;
    }
    case 2:
    {
      values<0, true, false, true>(shape_value_dir0, u, u);
      __syncthreads();
      values<1, true, false, true>(shape_value_dir1, u, u);
      __syncthreads();

      gradients<0, true, false, false>(co_shape_gradient_dir0, u, grad_u[0]);
      gradients<1, true, false, false>(co_shape_gradient_dir1, u, grad_u[1]);

      break;
    }
    case 3:
    {
      values<0, true, false, true>(shape_value_dir0, u, u);
      __syncthreads();
      values<1, true, false, true>(shape_value_dir1, u, u);
      __syncthreads();
      values<2, true, false, true>(shape_value_dir2, u, u);
      __syncthreads();

      gradients<0, true, false, false>(co_shape_gradient_dir0, u, grad_u[0]);
      gradients<1, true, false, false>(co_shape_gradient_dir1, u, grad_u[1]);
      gradients<2, true, false, false>(co_shape_gradient_dir2, u, grad_u[2]);

      break;
    }
    default:
    {
    }
  }
}



template<int dim, int fe_degree, int n_q_points_1d, typename Number>
template<bool add>
inline __device__ void
EvaluatorTensorProduct<evaluate_face, dim, fe_degree, n_q_points_1d, Number>::integrate_gradient(
  Number * u,
  Number * grad_u[dim])
{
  constexpr unsigned int n_q_points_2d = n_q_points_1d * n_q_points_1d;

  const unsigned int shift = (face_number & 1) * n_q_points_2d;

  Number * shape_value_dir0 =
    face_number / 2 == 0 ? shape_values + n_q_points_2d + shift : shape_values;

  Number * shape_value_dir1 =
    face_number / 2 == 1 ? shape_values + n_q_points_2d + shift : shape_values;

  Number * shape_value_dir2 =
    face_number / 2 == 2 ? shape_values + n_q_points_2d + shift : shape_values;

  Number * shape_gradient_dir0 =
    face_number / 2 == 0 ? shape_gradients + n_q_points_2d + shift : shape_gradients;

  Number * shape_gradient_dir1 =
    face_number / 2 == 1 ? shape_gradients + n_q_points_2d + shift : shape_gradients;

  Number * shape_gradient_dir2 =
    face_number / 2 == 2 ? shape_gradients + n_q_points_2d + shift : shape_gradients;

  switch(dim)
  {
    case 1:
    {
      gradients<0, false, add, false>(shape_gradient_dir0, grad_u[dim], u);

      break;
    }
    case 2:
    {
      gradients<0, false, false, true>(shape_gradient_dir0, grad_u[0], grad_u[0]);
      values<0, false, false, true>(shape_value_dir0, grad_u[1], grad_u[1]);

      __syncthreads();

      values<1, false, add, false>(shape_value_dir1, grad_u[0], u);
      __syncthreads();
      gradients<1, false, true, false>(shape_gradient_dir1, grad_u[1], u);

      break;
    }
    case 3:
    {
      gradients<0, false, false, true>(shape_gradient_dir0, grad_u[0], grad_u[0]);
      values<0, false, false, true>(shape_value_dir0, grad_u[1], grad_u[1]);
      values<0, false, false, true>(shape_value_dir0, grad_u[2], grad_u[2]);

      __syncthreads();

      values<1, false, false, true>(shape_value_dir1, grad_u[0], grad_u[0]);
      gradients<1, false, false, true>(shape_gradient_dir1, grad_u[1], grad_u[1]);
      values<1, false, false, true>(shape_value_dir1, grad_u[2], grad_u[2]);

      __syncthreads();

      values<2, false, add, false>(shape_value_dir2, grad_u[0], u);
      __syncthreads();
      values<2, false, true, false>(shape_value_dir2, grad_u[1], u);
      __syncthreads();
      gradients<2, false, true, false>(shape_gradient_dir2, grad_u[2], u);

      break;
    }
    default:
    {
    }
  }
}


template<int dim, int fe_degree, int n_q_points_1d, typename Number>
inline __device__ void
EvaluatorTensorProduct<evaluate_face, dim, fe_degree, n_q_points_1d, Number>::
  integrate_value_and_gradient(Number * u, Number * grad_u[dim])
{
  constexpr unsigned int n_q_points_2d = n_q_points_1d * n_q_points_1d;

  const unsigned int shift = (face_number & 1) * n_q_points_2d;

  Number * shape_value_dir0 =
    face_number / 2 == 0 ? shape_values + n_q_points_2d + shift : shape_values;

  Number * shape_value_dir1 =
    face_number / 2 == 1 ? shape_values + n_q_points_2d + shift : shape_values;

  Number * shape_value_dir2 =
    face_number / 2 == 2 ? shape_values + n_q_points_2d + shift : shape_values;

  // TODO:
  Number * co_shape_gradient_dir0 =
    face_number / 2 == 0 ? co_shape_gradients + shift : co_shape_gradients;

  Number * co_shape_gradient_dir1 =
    face_number / 2 == 1 ? co_shape_gradients + shift : co_shape_gradients;

  Number * co_shape_gradient_dir2 =
    face_number / 2 == 2 ? co_shape_gradients + shift : co_shape_gradients;

  switch(dim)
  {
    case 1:
    {
      gradients<0, false, true, false>(co_shape_gradient_dir0, grad_u[0], u);
      __syncthreads();

      values<0, false, false, true>(shape_value_dir0, u, u);

      break;
    }
    case 2:
    {
      gradients<1, false, true, false>(co_shape_gradient_dir1, grad_u[1], u);
      __syncthreads();
      gradients<0, false, true, false>(co_shape_gradient_dir0, grad_u[0], u);
      __syncthreads();

      values<1, false, false, true>(shape_value_dir1, u, u);
      __syncthreads();
      values<0, false, false, true>(shape_value_dir0, u, u);
      __syncthreads();

      break;
    }
    case 3:
    {
      gradients<2, false, true, false>(co_shape_gradient_dir2, grad_u[2], u);
      __syncthreads();
      gradients<1, false, true, false>(co_shape_gradient_dir1, grad_u[1], u);
      __syncthreads();
      gradients<0, false, true, false>(co_shape_gradient_dir0, grad_u[0], u);
      __syncthreads();

      values<2, false, false, true>(shape_value_dir2, u, u);
      __syncthreads();
      values<1, false, false, true>(shape_value_dir1, u, u);
      __syncthreads();
      values<0, false, false, true>(shape_value_dir0, u, u);
      __syncthreads();

      break;
    }
    default:
    {
    }
  }
}

} // namespace CUDAWrappers
} // namespace ExaDG

#endif // INCLUDE_EXADG_MATRIX_FREE_CUDA_TENSOR_PRODUCT_KERNEL_CUH_
