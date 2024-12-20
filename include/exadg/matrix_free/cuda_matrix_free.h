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

#ifndef INCLUDE_FUNCTIONALITIES_CUDA_MATRIX_FREE_CUH_
#define INCLUDE_FUNCTIONALITIES_CUDA_MATRIX_FREE_CUH_

// deal.II
#include <deal.II/base/config.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>

// ExaDG
#include <exadg/matrix_free/matrix_free_data.h>

namespace ExaDG
{
namespace CUDAWrappers
{
// Forward declaration
template<int dim, typename Number>
class ReinitHelper;

template<int dim, typename Number>
class MatrixFree : public dealii::Subscriptor
{
public:
  /**
   * Structure which is passed to the kernel. It is used to pass all the
   * necessary information from the CPU to the GPU.
   */
  struct Data
  {
    /**
     * Pointer to the quadrature points.
     */
    dealii::Point<dim, Number> * q_points;

    /**
     * Pointer to the face quadrature points.
     */
    dealii::Point<dim, Number> * face_q_points;

    /**
     * Map the position in the local vector to the position in the global
     * vector.
     */
    dealii::types::global_dof_index * local_to_global;

    /**
     * For faces, map the position in the local vector to the position
     * in the global vector.
     */
    dealii::types::global_dof_index * face_local_to_global;

    /**
     * Pointer to the cell inverse Jacobian.
     */
    Number * inv_jacobian;

    /**
     * Pointer to the cell Jacobian times the weights.
     */
    Number * JxW;

    /**
     * Pointer to the face inverse Jacobian.
     */
    Number * face_inv_jacobian;

    /**
     * Pointer to the face Jacobian times the weights.
     */
    Number * face_JxW;

    /**
     * Pointer to the unit normal vector on a face.
     */
    Number * normal_vector;

    /**
     * Pointer to the 1D shape info.
     */
    Number * cell_face_shape_values;
    Number * cell_face_shape_gradients;
    Number * cell_face_co_shape_gradients;

    /**
     * Pointer to the face direction.
     */
    unsigned int * face_number;

    /**
     * Pointer to the boundary id.
     */
    unsigned int * boundary_id;

    /**
     * Pointer to the cell to faces mapping.
     */
    dealii::types::global_dof_index * cell2face_id;

    /**
     * Number of objects.
     */
    unsigned int n_objs;

    /**
     * Number of cells.
     */
    unsigned int n_cells;

    /**
     * Number of faces.
     */
    unsigned int n_faces;

    /**
     * Number of inner faces.
     */
    unsigned int n_inner_faces;

    /**
     * Length of the padding.
     */
    unsigned int padding_length;

    /**
     * Length of the face padding.
     */
    unsigned int face_padding_length;

    /**
     * Row start (including padding).
     */
    unsigned int row_start;

    /*
     * Type of cells and faces.
     * NOTE: All objects have the same type now.
     */
    dealii::internal::MatrixFreeFunctions::GeometryType geometry_type;
  };

  /**
   * Default constructor.
   */
  MatrixFree();

  /**
   * Destructor.
   */
  ~MatrixFree();

  /**
   * Extracts the information needed to perform loops over cells. The
   * DoFHandler and AffineConstraints objects describe the layout of
   * degrees of freedom, the DoFHandler and the mapping describe the
   * transformation from unit to real cell, and the finite element
   * underlying the DoFHandler together with the quadrature formula
   * describe the local operations.
   *
   * TODO: several dof_handler, constraints and quad.
   */
  void
  reinit(const dealii::Mapping<dim> &                                     mapping,
         const dealii::DoFHandler<dim> &                                  dof_handler,
         const dealii::AffineConstraints<Number> &                        constraints,
         const dealii::Quadrature<dim> &                                  quad,
         const typename dealii::MatrixFree<dim, Number>::AdditionalData & additional_data);

  template<typename IteratorFiltersType>
  void
  reinit(const dealii::Mapping<dim> &                                     mapping,
         const dealii::DoFHandler<dim> &                                  dof_handler,
         const dealii::AffineConstraints<Number> &                        constraints,
         const dealii::Quadrature<dim> &                                  quad,
         const IteratorFiltersType &                                      iterator_filter,
         const typename dealii::MatrixFree<dim, Number>::AdditionalData & additional_data);

  /**
   * Return the multigrid level.
   */
  unsigned int
  get_mg_level() const;

  /**
   * Return the finite element degree.
   */
  unsigned int
  get_fe_degree() const;

  /**
   * Return the length of the padding.
   */
  unsigned int
  get_padding_length() const;

  /**
   * Return the length of the face padding.
   */
  unsigned int
  get_face_padding_length() const;

  /**
   * Return the Data structure associated with @p color.
   */
  Data
  get_cell_face_data(unsigned int color) const;

  /**
   * Return the Data structure associated with @p color.
   */
  Data
  get_boundary_face_data(unsigned int color) const;

  // clang-format off
  /**
   * This method runs the loop over all cells and apply the local operation on
   * each element in parallel. @p cell_operation is a functor which is 
   * applied on each color.
   * NOTE: loop over both cells and faces
   *
   * @p func needs to define
   * \code
   * __device__ void operator()(
   *   const unsigned int                            cell,
   *   const typename MatrixFree<dim, Number>::Data *gpu_data,
   *   SharedData<dim, Number> *                     shared_data,
   *   const Number *                                src,
   *   Number *                                      dst) const;
   *   static const unsigned int n_dofs_1d;
   *   static const unsigned int n_local_dofs;
   *   static const unsigned int n_q_points;
   * \endcode
   */
  // clang-format on
  template<typename Functor, typename VectorType>
  void
  cell_loop(const Functor & cell_operation, const VectorType & src, VectorType & dst) const;

  template<typename Functor, typename VectorType>
  void
  boundary_face_loop(const Functor &    face_operation,
                     const VectorType & src,
                     VectorType &       dst) const;

  /**
   * This method runs the loop over all cells and faces and apply the local operation on
   * each element in parallel. @p cell_operation is a functor which is applied
   * on each color. As opposed to the other variants that only runs a function
   * on cells or faces, this method runs on both cells and faces in a cell-wise manner.
   */
  template<typename Functor, typename VectorType>
  void
  cell_face_loop(const Functor &    cell_face_operation,
                 const VectorType & src,
                 VectorType &       dst) const;

  /**
   * This method runs the loop over all cells and apply the local operation on
   * each element in parallel. This function is very similar to cell_loop()
   * but it uses a simpler functor.
   *
   * @p func needs to define
   * \code
   *  __device__ void operator()(
   *    const unsigned int                            cell,
   *    const typename MatrixFree<dim, Number>::Data *gpu_data);
   * static const unsigned int n_dofs_1d;
   * static const unsigned int n_local_dofs;
   * static const unsigned int n_q_points;
   * \endcode
   */
  template<typename Functor>
  void
  evaluate_coefficients(Functor func) const;

  /**
   * Initialize a distributed vector. The local elements correspond to the
   * locally owned degrees of freedom and the ghost elements correspond to the
   * (additional) locally relevant dofs.
   * TODO: several DoFHandler objects
   */
  void
  initialize_dof_vector(
    dealii::LinearAlgebra::distributed::Vector<Number, dealii::MemorySpace::CUDA> & vec,
    const unsigned int dof_handler_index = 0) const;

  /**
   * Return the partitioner that represents the locally owned data and the
   * ghost indices where access is needed to for the cell loop. The
   * partitioner is constructed from the locally owned dofs and ghost dofs
   * given by the respective fields. If you want to have specific information
   * about these objects, you can query them with the respective access
   * functions. If you just want to initialize a (parallel) vector, you should
   * usually prefer this data structure as the data exchange information can
   * be reused from one vector to another.
   * TODO: several DoFHandler objects
   */
  const std::shared_ptr<const dealii::Utilities::MPI::Partitioner> &
  get_vector_partitioner() const;

  /**
   * Free all the memory allocated.
   */
  void
  free();

  /**
   * Return the DoFHandler.
   */
  const dealii::DoFHandler<dim> &
  get_dof_handler() const;

private:
  template<typename IteratorFiltersType>
  void
  internal_reinit(const dealii::Mapping<dim> &                                     mapping,
                  const dealii::DoFHandler<dim> &                                  dof_handler,
                  const dealii::AffineConstraints<Number> &                        constraints,
                  const dealii::Quadrature<dim> &                                  quad,
                  const IteratorFiltersType &                                      iterator_filter,
                  std::shared_ptr<const MPI_Comm>                                  comm,
                  const typename dealii::MatrixFree<dim, Number>::AdditionalData & additional_data);

  /**
   * Helper function. Loop over all the cells and apply the functor on each
   * element in parallel. This function is used when MPI is not used.
   */
  template<typename Functor, typename VectorType>
  void
  serial_cell_loop(const Functor & func, const VectorType & src, VectorType & dst) const;

  /**
   * Helper function. Loop over all the cells and apply the functor on each
   * element in parallel. This function is used when MPI is used.
   */
  template<typename Functor, typename VectorType>
  void
  distributed_cell_loop(const Functor & func, const VectorType & src, VectorType & dst) const;

  /**
   * Stored the level of the mesh to be worked on.
   */
  unsigned int mg_level;

  /**
   *  Overlap MPI communications with computation. This requires CUDA-aware
   *  MPI and use_coloring must be false.
   */
  bool overlap_communication_computation;

  /**
   * Total number of degrees of freedom.
   */
  dealii::types::global_dof_index n_dofs;

  /**
   * Degree of the finite element used.
   */
  unsigned int fe_degree;

  /**
   * Number of degrees of freedom per cell.
   */
  unsigned int dofs_per_cell;

  /**
   * Number of degrees of freedom per face.
   */
  unsigned int dofs_per_face;

  /**
   * Number of quadrature points per cell.
   */
  unsigned int q_points_per_cell;

  /**
   * Number of quadrature points per face.
   */
  unsigned int q_points_per_face;

  /**
   * Number of colors produced by the graph coloring algorithm.
   */
  unsigned int n_colors;

  /**
   * Number of cells in each color.
   */
  std::vector<unsigned int> n_cells;

  /**
   * Number of faces in each color.
   */
  std::vector<unsigned int> n_faces;

  /**
   * Number of inner faces in each color.
   */
  std::vector<unsigned int> n_inner_faces;

  /**
   * Number of boundary faces in each color.
   */
  std::vector<unsigned int> n_boundary_faces;

  /**
   * Vector of pointers to the quadrature points associated to the cells of
   * each color.
   */
  std::vector<dealii::Point<dim, Number> *> q_points;

  /**
   * Vector of pointers to the face quadrature points associated to the cells
   * of each color.
   */
  std::vector<dealii::Point<dim, Number> *> face_q_points;

  /**
   * Map the position in the local vector to the position in the global
   * vector.
   */
  std::vector<dealii::types::global_dof_index *> local_to_global;

  /**
   * Map the cell id to faces.
   */
  std::vector<dealii::types::global_dof_index *> cell2face_id;

  /**
   * For faces, map the position in the local vector to the position in the
   * global vector.
   */
  std::vector<dealii::types::global_dof_index *> face_local_to_global;

  /**
   * Vector of pointer to the cell inverse Jacobian associated to the cells of
   * each color.
   */
  std::vector<Number *> inv_jacobian;

  /**
   * Vector of pointer to the cell Jacobian times the weights associated to
   * the cells of each color.
   */
  std::vector<Number *> JxW;

  /**
   * Vector of pointer to the face inverse Jacobian associated to the cells of
   * each color.
   */
  std::vector<Number *> face_inv_jacobian;

  /**
   * Vector of pointer to the face Jacobian times the weights associated to
   * the cells of each color.
   */
  std::vector<Number *> face_JxW;

  /**
   * Vector of pointer to the unit normal vector on a face associated to the
   * cells of each color.
   */
  std::vector<Number *> normal_vector;

  /**
   * Vector of pointer to the face direction.
   */
  std::vector<unsigned int *> face_number;

  /**
   * Vector of pointer to the boundary id.
   */
  std::vector<unsigned int *> boundary_id;

  /**
   * Pointer to the 1D shape info.
   */
  Number * cell_face_shape_values;
  Number * cell_face_shape_gradients;
  Number * cell_face_co_shape_gradients;

  /**
   * Grid dimensions associated to the different colors. The grid dimensions
   * are used to launch the CUDA kernels.
   */
  std::vector<dim3> grid_dim;

  /**
   * Grid dimensions associated to the different colors. The grid dimensions
   * are used to launch the CUDA kernels.
   */
  std::vector<dim3> grid_dim_inner_face;

  /**
   * Grid dimensions associated to the different colors. The grid dimensions
   * are used to launch the CUDA kernels.
   */
  std::vector<dim3> grid_dim_boundary_face;

  /**
   * Block dimensions associated to the different colors. The block dimensions
   * are used to launch the CUDA kernels.
   */
  std::vector<dim3> block_dim;

  /**
   * Block dimensions associated to the different colors. The block dimensions
   * are used to launch the CUDA kernels.
   */
  std::vector<dim3> block_dim_inner_face;

  /**
   * Block dimensions associated to the different colors. The block dimensions
   * are used to launch the CUDA kernels.
   */
  std::vector<dim3> block_dim_boundary_face;

  /**
   * Shared pointer to a Partitioner for distributed Vectors used in
   * cell_loop. When MPI is not used the pointer is null.
   */
  std::shared_ptr<const dealii::Utilities::MPI::Partitioner> partitioner;

  /**
   * Cells per block (determined by the function cells_per_block_shmem() ).
   */
  unsigned int cells_per_block;

  /**
   * Boundary faces per block (determined by the function todo() ).
   */
  unsigned int boundary_faces_per_block;

  /**
   * Inner faces per block (determined by the function todo() ).
   */
  unsigned int inner_faces_per_block;

  /**
   * Length of the padding (closest power of two larger than or equal to
   * the number of thread).
   */
  unsigned int padding_length;

  /**
   * Length of the face padding (closest power of two larger than or equal to
   * the number of thread).
   */
  unsigned int face_padding_length;

  /**
   * Row start of each color.
   */
  std::vector<unsigned int> row_start;

  /*
   * Type of cells and faces.
   * NOTE: All objects have the same type now.
   */
  dealii::internal::MatrixFreeFunctions::GeometryType geometry_type;

  /**
   * Pointer to the DoFHandler associated with the object.
   */
  const dealii::DoFHandler<dim> * dof_handler;

  friend class ReinitHelper<dim, Number>;
};

/**
 * Structure to pass the shared memory into a general user function.
 */
template<int dim, typename Number>
struct SharedData
{
  /**
   * Constructor.
   */
  __host__ __device__
  SharedData(Number *           data,
             const unsigned int local_cell,
             const unsigned int n_dofs_2d,
             const unsigned int n_local_dofs,
             const unsigned int cells_per_block)
  {
    values = data + local_cell * n_local_dofs;
    for(unsigned int d = 0; d < dim; ++d)
      gradients[d] = data + (cells_per_block + local_cell * dim + d) * n_local_dofs;

    shape_values       = data + cells_per_block * n_local_dofs * (dim + 1);
    shape_gradients    = shape_values + 3 * n_dofs_2d;
    co_shape_gradients = shape_gradients + 3 * n_dofs_2d;
  }

  /**
   * Shared memory for dof and quad values.
   */
  Number * values;

  /**
   * Shared memory for computed gradients in reference coordinate system.
   * The gradient in each direction is saved in a struct-of-array
   * format, i.e. first, all gradients in the x-direction come...
   */
  Number * gradients[dim];

  /*
   * Shared memory for 1D shape values.
   */
  Number * shape_values;

  /*
   * Shared memory for 1D shape gradients.
   */
  Number * shape_gradients;

  /*
   * Shared memory for 1D co shape gradients.
   */
  Number * co_shape_gradients;
};

// This function determines the number of cells per block, possibly at compile
// time (by virtue of being 'constexpr')
// TODO: this function should be rewritten using meta-programming
__host__ __device__ constexpr unsigned int
cells_per_block_shmem(int dim, int fe_degree)
{
  return 1;

  constexpr int warp_size = 32;

  return dim==2 ? (fe_degree==1 ? warp_size :    // 128
                     fe_degree==2 ? warp_size/4 :  //  72
                     fe_degree==3 ? warp_size/8 :  //  64
                     fe_degree==4 ? warp_size/8 :  // 100
                     1) :
           dim==3 ? (fe_degree==1 ? 1 :  //  
                     fe_degree==2 ? 8 : // 1 < 2 < 4 < 8 > 16  
                     fe_degree==3 ? 2 : // 1 2 4 8 same perf
                     1) : 1;
}

// TODO:
__host__ __device__ constexpr unsigned int
faces_per_block_shmem(int dim, int fe_degree)
{
  return 1;

  constexpr int warp_size = 32;

  return dim==2 ? (fe_degree==1 ? warp_size :    // 128
                     fe_degree==2 ? warp_size/4 :  //  72
                     fe_degree==3 ? warp_size/8 :  //  64
                     fe_degree==4 ? warp_size/8 :  // 100
                     1) :
           dim==3 ? (fe_degree==1 ? warp_size/4 :  //  64
                     fe_degree==2 ? warp_size/16 : //  54
                     1) : 1;
}


/*----------------------------- Inline functions -----------------------------*/
template<int dim, typename Number>
inline const std::shared_ptr<const dealii::Utilities::MPI::Partitioner> &
MatrixFree<dim, Number>::get_vector_partitioner() const
{
  return partitioner;
}

template<int dim, typename Number>
inline const dealii::DoFHandler<dim> &
MatrixFree<dim, Number>::get_dof_handler() const
{
  Assert(dof_handler != nullptr, dealii::ExcNotInitialized());

  return *dof_handler;
}

} // namespace CUDAWrappers
} // namespace ExaDG

#include <exadg/matrix_free/cuda_matrix_free.templates.cuh>

#endif /* INCLUDE_FUNCTIONALITIES_CUDA_MATRIX_FREE_CUH_ */
