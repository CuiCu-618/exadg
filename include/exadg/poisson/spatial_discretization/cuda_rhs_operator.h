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

#ifndef INCLUDE_OPERATORS_LAPLACE_DG_CUDA_RHS_OPERATOR
#define INCLUDE_OPERATORS_LAPLACE_DG_CUDA_RHS_OPERATOR

#include <exadg/functions_and_boundary_conditions/evaluate_functions.h>
#include <exadg/matrix_free/cuda_matrix_free.h>
#include <exadg/matrix_free/integrators.h>
#include <exadg/operators/mapping_flags.h>
#include <exadg/matrix_free/cuda_fe_evaluation.cuh>

namespace ExaDG
{
namespace CUDAWrappers
{
namespace Operators
{
template<int dim>
struct RHSKernelData
{
  std::shared_ptr<dealii::Function<dim>> f;
};

template<int dim, typename Number, int n_components = 1>
class RHSKernel
{
private:
  static unsigned int const rank =
    (n_components == 1) ? 0 : ((n_components == dim) ? 1 : dealii::numbers::invalid_unsigned_int);

  double const FREQUENCY = 3.0 * dealii::numbers::PI;

public:
  void
  reinit(RHSKernelData<dim> const & data_in) const
  {
    data = data_in;
  }

  static MappingFlags
  get_mapping_flags()
  {
    MappingFlags flags;

    flags.cells = dealii::update_JxW_values |
                  dealii::update_quadrature_points; // q-points due to rhs function f

    // no face integrals

    return flags;
  }

  /*
   * Volume flux, i.e., the term occurring in the volume integral
   */
  __device__ inline Number
  get_volume_flux(const dealii::Point<dim, Number> & q_point, Number const &) const
  {
    Number val = FREQUENCY * FREQUENCY * dim;
    for(unsigned int d = 0; d < dim; ++d)
      val *= sin(FREQUENCY * q_point[d]);

    return val;
  }

private:
  mutable RHSKernelData<dim> data;
};

} // namespace Operators


template<int dim>
struct RHSOperatorData
{
  RHSOperatorData() : dof_index(0), quad_index(0)
  {
  }

  unsigned int dof_index;
  unsigned int quad_index;

  Operators::RHSKernelData<dim> kernel_data;
};

template<int dim, typename Number, int n_components = 1>
class RHSOperator
{
private:
  typedef dealii::LinearAlgebra::distributed::Vector<Number, dealii::MemorySpace::CUDA> VectorType;

  typedef RHSOperator<dim, Number, n_components> This;

public:
  /*
   * Constructor.
   */
  RHSOperator() : matrix_free(nullptr), time(0.0)
  {
  }

  /*
   * Initialization.
   */
  void
  initialize(CUDAWrappers::MatrixFree<dim, Number> const & matrix_free_in,
             RHSOperatorData<dim> const &                  data_in)
  {
    this->matrix_free = &matrix_free_in;
    this->data        = data_in;
    this->fe_degree   = matrix_free_in.get_fe_degree();

    kernel.reinit(data.kernel_data);
  }

  /*
   * Evaluate operator and overwrite dst-vector.
   */
  void
  evaluate(VectorType & dst, double const evaluation_time) const
  {
    dst = 0;
    evaluate_add(dst, evaluation_time);
  }

  /*
   * Evaluate operator and add to dst-vector.
   */
  void
  evaluate_add(VectorType & dst, double const evaluation_time) const
  {
    this->time = evaluation_time;

    switch(fe_degree)
    {
        // clang-format off
      case  1: do_evaluate_add<1>(dst); break;
      case  2: do_evaluate_add<2>(dst); break;
      case  3: do_evaluate_add<3>(dst); break;
      case  4: do_evaluate_add<4>(dst); break;
      case  5: do_evaluate_add<5>(dst); break;
      case  6: do_evaluate_add<6>(dst); break;
      case  7: do_evaluate_add<7>(dst); break;
      case  8: do_evaluate_add<8>(dst); break;
      case  9: do_evaluate_add<9>(dst); break;
      case 10: do_evaluate_add<10>(dst); break;
      default:
        AssertThrow(false, dealii::ExcNotImplemented("Only degrees 1 through 10 implemented."));
        // clang-format on
    }
  }

  template<int fe_degree>
  void
  do_evaluate_add(VectorType & dst) const
  {
    VectorType tmp(dst.get_partitioner());

    LocalRHSOperator<fe_degree> rhs_op;
    matrix_free->cell_loop(rhs_op, tmp, dst);
  }

  template<int fe_degree>
  class LocalRHSOperator
  {
  public:
    static const unsigned int n_dofs_1d    = fe_degree + 1;
    static const unsigned int n_local_dofs = dealii::Utilities::pow(fe_degree + 1, dim);
    static const unsigned int n_q_points   = dealii::Utilities::pow(fe_degree + 1, dim);

    // TODO: less shared memory usage.
    // Rearrange shared_mem pointer needed
    static const unsigned int shared_mem =
      (n_local_dofs * (dim + 1) + 7 * n_dofs_1d * n_dofs_1d) * sizeof(Number);

    static const unsigned int cells_per_block = CUDAWrappers::cells_per_block_shmem(dim, fe_degree);
    static const unsigned int n_dofs_z        = dim == 3 ? fe_degree + 1 : 1;

    using value_type = cuda::std::array<Number, n_dofs_z>;

    double const FREQUENCY = 3.0 * dealii::numbers::PI;

    LocalRHSOperator()
    {
    }

    __device__ void
    operator()(const unsigned int                                           cell,
               const typename CUDAWrappers::MatrixFree<dim, Number>::Data * gpu_data,
               CUDAWrappers::SharedData<dim, Number> *                      shared_data,
               const Number *                                               src,
               Number *                                                     dst) const
    {
      (void)src;

      CUDAWrappers::FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> fe_eval(cell,
                                                                                   gpu_data,
                                                                                   shared_data);

      value_type val{};
      for(unsigned int i = 0; i < n_dofs_z; ++i)
      {
        const auto index =
          i * n_dofs_1d * n_dofs_1d + ExaDG::CUDAWrappers::q_point_id_in_cell<dim, n_dofs_1d>();
        const auto point =
          ExaDG::CUDAWrappers::get_quadrature_point<dim, Number, n_dofs_1d>(cell, gpu_data, index);

        val[i] = FREQUENCY * FREQUENCY * dim;
        for(unsigned int d = 0; d < dim; ++d)
          val[i] *= sin(FREQUENCY * point[d]);
      }

      fe_eval.submit_value(val);
      fe_eval.integrate(true, false);

      fe_eval.distribute_local_to_global(dst);
    }
  };

private:
  CUDAWrappers::MatrixFree<dim, Number> const * matrix_free;

  RHSOperatorData<dim> data;

  mutable double time;

  unsigned int fe_degree;

  Operators::RHSKernel<dim, Number, n_components> kernel;
};

} // namespace CUDAWrappers
} // namespace ExaDG

#endif
