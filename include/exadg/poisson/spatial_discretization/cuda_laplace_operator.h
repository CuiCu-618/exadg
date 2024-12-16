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

#ifndef CUDA_LAPLACE_OPERATOR_H
#define CUDA_LAPLACE_OPERATOR_H

#include <exadg/matrix_free/cuda_matrix_free.h>
#include <exadg/operators/interior_penalty_parameter.h>
#include <exadg/operators/operator_base.h>
#include <exadg/operators/operator_type.h>
#include <exadg/poisson/user_interface/boundary_descriptor.h>
#include <exadg/matrix_free/cuda_fe_evaluation.cuh>

#include <cuda/std/array>

namespace ExaDG
{
namespace Poisson
{
namespace CUDAWrappers
{
namespace Operators
{
struct LaplaceKernelData
{
  LaplaceKernelData() : IP_factor(1.0)
  {
  }

  double IP_factor;
};

template<int dim, typename Number, int n_components = 1>
class LaplaceKernel
{
private:
  typedef dealii::LinearAlgebra::distributed::Vector<Number, dealii::MemorySpace::CUDA> VectorType;

public:
  LaplaceKernel() : degree(1), tau(0.0)
  {
  }

  void
  reinit(ExaDG::CUDAWrappers::MatrixFree<dim, Number> const & matrix_free,
         LaplaceKernelData const &                            data_in,
         unsigned int const                                   dof_index)
  {
    (void)dof_index;

    data = data_in;

    dealii::FiniteElement<dim> const & fe = matrix_free.get_dof_handler().get_fe();
    degree                                = fe.degree;
  }

  IntegratorFlags
  get_integrator_flags(bool const is_dg) const
  {
    IntegratorFlags flags;

    flags.cell_evaluate  = dealii::EvaluationFlags::gradients;
    flags.cell_integrate = dealii::EvaluationFlags::gradients;

    if(is_dg)
    {
      flags.face_evaluate  = dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients;
      flags.face_integrate = dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients;
    }
    else
    {
      // evaluation of Neumann BCs for continuous elements
      flags.face_evaluate  = dealii::EvaluationFlags::nothing;
      flags.face_integrate = dealii::EvaluationFlags::values;
    }

    return flags;
  }

  static MappingFlags
  get_mapping_flags(bool const compute_interior_face_integrals,
                    bool const compute_boundary_face_integrals)
  {
    MappingFlags flags;

    flags.cells = dealii::update_gradients | dealii::update_JxW_values;

    if(compute_interior_face_integrals)
    {
      flags.inner_faces =
        dealii::update_gradients | dealii::update_JxW_values | dealii::update_normal_vectors;
    }

    if(compute_boundary_face_integrals)
    {
      flags.boundary_faces = dealii::update_gradients | dealii::update_JxW_values |
                             dealii::update_normal_vectors | dealii::update_quadrature_points;
    }

    return flags;
  }

private:
  LaplaceKernelData data;

  unsigned int degree;

  mutable Number tau;
};

} // namespace Operators

template<int rank, int dim>
struct LaplaceOperatorData : public OperatorBaseData
{
  LaplaceOperatorData() : OperatorBaseData(), quad_index_gauss_lobatto(0)
  {
  }

  Operators::LaplaceKernelData kernel_data;

  // continuous FE:
  // for DirichletCached boundary conditions, another quadrature rule
  // is needed to set the constrained DoFs.
  unsigned int quad_index_gauss_lobatto;

  std::shared_ptr<BoundaryDescriptor<rank, dim> const> bc;
};

template<int dim, typename Number, int n_components = 1>
class LaplaceOperator
{
private:
  static unsigned int const rank =
    (n_components == 1) ? 0 : ((n_components == dim) ? 1 : dealii::numbers::invalid_unsigned_int);

  typedef LaplaceOperator<dim, Number, n_components> This;

  typedef dealii::LinearAlgebra::distributed::Vector<Number, dealii::MemorySpace::CUDA> VectorType;

public:
  LaplaceOperator()
    : matrix_free(nullptr),
      time(0.0),
      is_mg(false),
      is_dg(true),
      level(dealii::numbers::invalid_unsigned_int),
      n_mpi_processes(0)
  {
  }

  void
  initialize(ExaDG::CUDAWrappers::MatrixFree<dim, Number> const & matrix_free,
             dealii::AffineConstraints<Number> const &            affine_constraints,
             LaplaceOperatorData<rank, dim> const &               data)
  {
    this->matrix_free = &matrix_free;
    this->constraint  = &affine_constraints;
    this->fe_degree   = this->matrix_free->get_fe_degree();
    operator_data     = data;

    dealii::DoFHandler<dim> const & dof_handler = this->matrix_free->get_dof_handler();

    is_dg           = (dof_handler.get_fe().dofs_per_vertex == 0);
    this->level     = this->matrix_free->get_mg_level();
    this->is_mg     = (this->level != dealii::numbers::invalid_unsigned_int);
    n_mpi_processes = dealii::Utilities::MPI::n_mpi_processes(dof_handler.get_communicator());

    kernel.reinit(matrix_free, data.kernel_data, data.dof_index);

    this->integrator_flags = kernel.get_integrator_flags(this->is_dg);
  }

  LaplaceOperatorData<rank, dim> const &
  get_data() const
  {
    return operator_data;
  }

  void
  set_time(double const t) const
  {
    this->time = t;
  }

  double
  get_time() const
  {
    return time;
  }

  unsigned int
  get_level() const
  {
    return level;
  }

  dealii::AffineConstraints<Number> const &
  get_affine_constraints() const
  {
    return *constraint;
  }

  ExaDG::CUDAWrappers::MatrixFree<dim, Number> const &
  get_matrix_free() const
  {
    return *this->matrix_free;
  }

  unsigned int
  get_dof_index() const
  {
    return this->data.dof_index;
  }

  unsigned int
  get_quad_index() const
  {
    return this->data.quad_index;
  }

  bool
  operator_is_singular() const
  {
    return this->data.operator_is_singular;
  }

  void
  vmult(VectorType & dst, VectorType const & src) const
  {
    this->apply(dst, src);
  }

  void
  vmult_add(VectorType & dst, VectorType const & src) const
  {
    this->apply_add(dst, src);
  }

  void
  vmult_interface_down(VectorType & dst, VectorType const & src) const
  {
    // TODO:
    // vmult(dst, src);
    AssertThrow(false, dealii::ExcMessage("Local refinement not implemented."));
  }

  void
  vmult_add_interface_up(VectorType & dst, VectorType const & src) const
  {
    // TODO:
    // vmult_add(dst, src);
    AssertThrow(false, dealii::ExcMessage("Local refinement not implemented."));
  }

  dealii::types::global_dof_index
  m() const
  {
    return n();
  }

  dealii::types::global_dof_index
  n() const
  {
    unsigned int dof_index = get_dof_index();

    return this->matrix_free->get_vector_partitioner(dof_index)->size();
  }

  Number
  el(unsigned int const, unsigned int const) const
  {
    AssertThrow(false, dealii::ExcMessage("Matrix-free does not allow for entry access"));
    return Number();
  }

  void
  initialize_dof_vector(VectorType & vector) const
  {
    unsigned int dof_index = get_dof_index();

    this->matrix_free->initialize_dof_vector(vector, dof_index);
  }

  void
  apply(VectorType & dst, VectorType const & src) const
  {
    dst = 0;
    this->apply_add(dst, src);
  }

  void
  apply_add(VectorType & dst, VectorType const & src) const
  {
    if(is_dg)
    {
      if(evaluate_face_integrals())
      {
        switch(fe_degree)
        {
            // clang-format off
          case  1: do_apply_add<1>(dst, src); break;
          case  2: do_apply_add<2>(dst, src); break;
          case  3: do_apply_add<3>(dst, src); break;
          case  4: do_apply_add<4>(dst, src); break;
          case  5: do_apply_add<5>(dst, src); break;
          case  6: do_apply_add<6>(dst, src); break;
          case  7: do_apply_add<7>(dst, src); break;
          case  8: do_apply_add<8>(dst, src); break;
          case  9: do_apply_add<9>(dst, src); break;
          case 10: do_apply_add<10>(dst, src); break;
          default:
            AssertThrow(false, dealii::ExcNotImplemented("Only degrees 1 through 10 implemented."));
            // clang-format on
        }
      }
      else
      {
      }
    }
  }

  template<int fe_degree>
  void
  do_apply_add(VectorType & dst, const VectorType & src) const
  {
    LocalCellFaceOperator<fe_degree> laplace_op;
    matrix_free->cell_loop(laplace_op, src, dst);
  }


  void
  rhs(VectorType & rhs) const
  {
    rhs = 0;
    this->rhs_add(rhs);
  }

  void
  rhs_add(VectorType & dst) const
  {
    VectorType tmp;
    tmp.reinit(dst, false);

    switch(fe_degree)
    {
        // clang-format off
      case  1: do_rhs_add<1>(dst); break;
      case  2: do_rhs_add<2>(dst); break;
      case  3: do_rhs_add<3>(dst); break;
      case  4: do_rhs_add<4>(dst); break;
      case  5: do_rhs_add<5>(dst); break;
      case  6: do_rhs_add<6>(dst); break;
      case  7: do_rhs_add<7>(dst); break;
      case  8: do_rhs_add<8>(dst); break;
      case  9: do_rhs_add<9>(dst); break;
      case 10: do_rhs_add<10>(dst); break;
      default:
        AssertThrow(false, dealii::ExcNotImplemented("Only degrees 1 through 10 implemented."));
        // clang-format on
    }

    dst.add(-1., tmp);
  }

  template<int fe_degree>
  void
  do_rhs_add(VectorType & dst) const
  {
    VectorType tmp(dst.get_partitioner());

    LocalBoundaryOperator<fe_degree> bd_op;
    matrix_free->boundary_face_loop(bd_op, tmp, dst);
  }


  void
  evaluate(VectorType & dst, VectorType const & src) const
  {
    dst = 0;
    evaluate_add(dst, src);
  }

  void
  evaluate_add(VectorType & dst, VectorType const & src) const
  {
    AssertThrow(false, dealii::ExcMessage("TODO: DG Laplace Operator."));
  }

  void
  calculate_diagonal(VectorType & diagonal) const
  {
    if(diagonal.size() == 0)
      matrix_free->initialize_dof_vector(diagonal);
    diagonal = 0;
    add_diagonal(diagonal);
  }

  void
  add_diagonal(VectorType & diagonal) const
  {
    if(is_dg && evaluate_face_integrals())
    {
      AssertThrow(false, dealii::ExcMessage("TODO: DG Laplace Operator."));
    }
  }

  bool
  evaluate_face_integrals() const
  {
    return (integrator_flags.face_evaluate != dealii::EvaluationFlags::nothing) ||
           (integrator_flags.face_integrate != dealii::EvaluationFlags::nothing);
  }

  // Some more functionality on top of what is provided by the base class.
  // This function evaluates the inhomogeneous boundary face integrals in DG where the
  // Dirichlet boundary condition is extracted from a dof vector instead of a dealii::Function<dim>.
  void
  rhs_add_dirichlet_bc_from_dof_vector(VectorType & dst, VectorType const & src) const;

  // continuous FE: This function sets the constrained Dirichlet boundary values.
  void
  set_constrained_values(VectorType & solution, double const time) const;


  template<int fe_degree>
  class LocalCellFaceOperator
  {
  public:
    static const unsigned int n_dofs_1d    = fe_degree + 1;
    static const unsigned int n_local_dofs = dealii::Utilities::pow(fe_degree + 1, dim) * 2;
    static const unsigned int n_q_points   = dealii::Utilities::pow(fe_degree + 1, dim) * 2;
    static const unsigned int shared_mem =
      (n_local_dofs * (dim + 1) + 9 * n_dofs_1d * n_dofs_1d) * sizeof(Number);

    static const unsigned int cells_per_block =
      1; // TODO: CUDAWrappers::cells_per_block_shmem(dim, fe_degree);

    static const unsigned int n_dofs_z = dim == 3 ? fe_degree + 1 : 1;

    using value_type = cuda::std::array<Number, n_dofs_z>;

    double const PENALTY_FACTOR = 1.0 * (fe_degree + 1) * (fe_degree + 1);

    OperatorType const operator_type;

    LocalCellFaceOperator(const OperatorType operator_type = OperatorType::homogeneous)
      : operator_type(operator_type)
    {
    }

    __device__ void
    operator()(const unsigned int                                                  cell,
               const typename ExaDG::CUDAWrappers::MatrixFree<dim, Number>::Data * gpu_data,
               ExaDG::CUDAWrappers::SharedData<dim, Number> *                      shared_data,
               const Number *                                                      src,
               Number *                                                            dst) const
    {
      ExaDG::CUDAWrappers::FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> fe_eval(
        cell, gpu_data, shared_data);

      fe_eval.read_dof_values(src);
      value_type dof_value_in = fe_eval.get_dof_value();

      fe_eval.evaluate(false, true);
      fe_eval.submit_gradient(fe_eval.get_gradient());
      fe_eval.integrate(false, true);

      value_type dof_value_out = fe_eval.get_dof_value();

      for(int f = 0; f < dim * 2; ++f)
      {
        auto face0 = gpu_data->cell2face_id[cell * dim * 2 * 2 + 2 * f];
        auto face1 = gpu_data->cell2face_id[cell * dim * 2 * 2 + 2 * f + 1];

        if(face0 + face1 == 0)
          continue;

        Number coe = 1.;
        if(face0 == face1 &&
           gpu_data->boundary_id[face0 - gpu_data->n_inner_faces] == 0) // Dirichlet
          coe = -1.;

        ExaDG::CUDAWrappers::FEFaceEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> phi_inner(
          face0, gpu_data, shared_data, true);
        ExaDG::CUDAWrappers::FEFaceEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> phi_outer(
          face1, gpu_data, shared_data, false);

        phi_inner.read_dof_values(src);
        phi_inner.evaluate(true, true);

        phi_outer.read_dof_values(src);
        phi_outer.evaluate(true, true);

        auto hi    = 0.5 * (fabs(phi_inner.inverse_length_normal_to_face()) +
                         fabs(phi_outer.inverse_length_normal_to_face()));
        auto sigma = hi * PENALTY_FACTOR;

        auto u_inner = phi_inner.get_value();
        auto u_outer = phi_outer.get_value();
        auto n_inner = phi_inner.get_normal_derivative();
        auto n_outer = phi_outer.get_normal_derivative();

        auto solution_jump             = u_inner - u_outer * coe;
        auto average_normal_derivative = (n_inner + n_outer * coe) * 0.5;
        auto test_by_value             = solution_jump * sigma - average_normal_derivative;

        phi_inner.submit_value(test_by_value);
        phi_outer.submit_value(test_by_value * -1.);
        phi_inner.submit_normal_derivative(solution_jump * -0.5);
        phi_outer.submit_normal_derivative(solution_jump * -0.5);

        phi_inner.integrate(true, true);
        dof_value_out = dof_value_out + phi_inner.get_dof_value();

        if(face0 == face1)
          continue;

        phi_outer.integrate(true, true);
        phi_outer.distribute_local_to_global(dst);
      }

      fe_eval.submit_dof_value(dof_value_out);
      fe_eval.distribute_local_to_global(dst);
    }
  };


  template<int fe_degree>
  class LocalBoundaryOperator
  {
  public:
    static const unsigned int n_dofs_1d    = fe_degree + 1;
    static const unsigned int n_local_dofs = dealii::Utilities::pow(fe_degree + 1, dim);
    static const unsigned int n_q_points   = dealii::Utilities::pow(fe_degree + 1, dim);
    static const unsigned int shared_mem =
      (n_local_dofs * (dim + 1) + 8 * n_dofs_1d * n_dofs_1d) * sizeof(Number);

    static const unsigned int cells_per_block =
      1; // TODO: CUDAWrappers::cells_per_block_shmem(dim, fe_degree);

    static const unsigned int n_dofs_z = dim == 3 ? fe_degree + 1 : 1;

    using value_type = cuda::std::array<Number, n_dofs_z>;

    double const PENALTY_FACTOR = 1.0 * (fe_degree + 1) * (fe_degree + 1);
    double const FREQUENCY      = 3.0 * dealii::numbers::PI;

    OperatorType const operator_type;

    LocalBoundaryOperator(const OperatorType operator_type = OperatorType::inhomogeneous)
      : operator_type(operator_type)
    {
    }

    __device__ void
    operator()(const unsigned int                                                  face,
               const typename ExaDG::CUDAWrappers::MatrixFree<dim, Number>::Data * gpu_data,
               ExaDG::CUDAWrappers::SharedData<dim, Number> *                      shared_data,
               const Number *                                                      src,
               Number *                                                            dst) const
    {
      ExaDG::CUDAWrappers::FEFaceEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> fe_eval(
        face, gpu_data, shared_data, true);

      if(gpu_data->boundary_id[face] == 0) // Dirichlet
      {
      }
      else // Neumann
      {
        auto face_number = gpu_data->face_number[face];

        value_type val = {};
        for(unsigned int i = 0; i < n_dofs_z; ++i)
        {
          const auto q_point_face =
            ExaDG::CUDAWrappers::compute_face_index<dim, n_dofs_1d>(face_number, i);
          const auto point =
            *(gpu_data->face_q_points + face * gpu_data->face_padding_length + q_point_face);

          val[i] = 1.;
          for(unsigned int d = 0; d < dim; ++d)
          {
            if(d == 0)
              val[i] *= FREQUENCY * cos(FREQUENCY * point[d]);
            else
              val[i] *= sin(FREQUENCY * point[d]);
          }
        }

        fe_eval.submit_value(val);
        fe_eval.integrate(true, false);

        fe_eval.distribute_local_to_global(dst);
      }
    }
  };



private:
  ExaDG::CUDAWrappers::MatrixFree<dim, Number> const * matrix_free;
  dealii::AffineConstraints<Number> const *            constraint;

  LaplaceOperatorData<rank, dim> operator_data;

  mutable double time;

  bool         is_mg;
  bool         is_dg;
  unsigned int fe_degree;
  unsigned int level;
  unsigned int n_mpi_processes;

  IntegratorFlags integrator_flags;

  Operators::LaplaceKernel<dim, Number, n_components> kernel;
};

} // namespace CUDAWrappers
} // namespace Poisson
} // namespace ExaDG

#endif
