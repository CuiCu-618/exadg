/*
 * DGNavierStokesDualSplittingXWall.h
 *
 *  Created on: Jul 7, 2016
 *      Author: krank
 */

#ifndef INCLUDE_DGNAVIERSTOKESDUALSPLITTINGXWALL_H_
#define INCLUDE_DGNAVIERSTOKESDUALSPLITTINGXWALL_H_

#include "DGNavierStokesDualSplitting.h"
#include <deal.II/base/utilities.h>

class FEParametersTauwN
{
public:

  FEParametersTauwN(FEParameters const & fe_param) :
    viscosity(fe_param.viscosity),
    cs(fe_param.cs),
    ml(fe_param.ml),
    variabletauw(fe_param.variabletauw),
    dtauw(fe_param.dtauw),
    max_wdist_xwall(fe_param.max_wdist_xwall),
    wdist(fe_param.wdist)
  {
  }

  double const & viscosity;
  double const & cs;
  double const & ml;
  bool const & variabletauw;
  double const & dtauw;
  double const & max_wdist_xwall;
  parallel::distributed::Vector<double> const &wdist;
  parallel::distributed::Vector<double> tauw;
};


template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
class DGNavierStokesDualSplittingXWall : public DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>
{
public:
  typedef typename DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::value_type value_type;
  typedef typename DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::FEFaceEval_Velocity_Velocity_linear FEFaceEval_Velocity_Velocity_linear;
  typedef typename DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::FEEval_Velocity_Velocity_linear FEEval_Velocity_Velocity_linear;

  DGNavierStokesDualSplittingXWall(parallel::distributed::Triangulation<dim> const &triangulation,
                                   InputParameters const                           &parameter)
    :
      DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>(triangulation,parameter),
      fe_wdist(QGaussLobatto<1>(1+1)),
      dof_handler_wdist(triangulation),
      fe_param_n(this->fe_param)
  {
    this->fe_u.reset(new FESystem<dim>(FE_DGQArbitraryNodes<dim>(QGaussLobatto<1>(fe_degree+1)),dim,FE_DGQArbitraryNodes<dim>(QGaussLobatto<1>(fe_degree_xwall+1)),dim));
  }

  virtual ~DGNavierStokesDualSplittingXWall(){}

  virtual void setup (const std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator> > periodic_face_pairs,
              std::set<types::boundary_id> dirichlet_bc_indicator,
              std::set<types::boundary_id> neumann_bc_indicator);

  void update_tauw(parallel::distributed::Vector<value_type> &velocity);

  void precompute_inverse_mass_matrix();

  void xwall_projection();
private:
  virtual void create_dofs();

  virtual void data_reinit(typename MatrixFree<dim,value_type>::AdditionalData & additional_data);

  virtual void push_back_constraint_matrix(std::vector<const ConstraintMatrix *> & constraint_matrix_vec);

  void init_wdist();

  void calculate_wall_shear_stress(const parallel::distributed::Vector<value_type> &src,
                                         parallel::distributed::Vector<value_type> &dst);

  void local_rhs_dummy (const MatrixFree<dim,value_type>                &,
                        parallel::distributed::Vector<value_type>      &,
                        const parallel::distributed::Vector<value_type>  &,
                        const std::pair<unsigned int,unsigned int>           &) const;

  void local_rhs_dummy_face (const MatrixFree<dim,value_type>                 &,
                             parallel::distributed::Vector<value_type>      &,
                             const parallel::distributed::Vector<value_type>  &,
                             const std::pair<unsigned int,unsigned int>          &) const;

  void local_rhs_wss_boundary_face (const MatrixFree<dim,value_type>             &data,
                                    parallel::distributed::Vector<value_type>    &dst,
                                    const parallel::distributed::Vector<value_type>  &src,
                                    const std::pair<unsigned int,unsigned int>          &face_range) const;

  void local_rhs_normalization_boundary_face (const MatrixFree<dim,value_type>             &data,
                                              parallel::distributed::Vector<value_type>    &dst,
                                              const parallel::distributed::Vector<value_type>  &,
                                              const std::pair<unsigned int,unsigned int>   &face_range) const;

  // inverse mass matrix velocity
  void local_precompute_mass_matrix(const MatrixFree<dim,value_type>                &data,
                                    parallel::distributed::Vector<value_type>    &,
                                    const parallel::distributed::Vector<value_type>  &,
                                    const std::pair<unsigned int,unsigned int>          &cell_range);

  // inverse mass matrix velocity
  void local_project_xwall(const MatrixFree<dim,value_type>                &data,
                      parallel::distributed::Vector<value_type>    &dst,
                      const parallel::distributed::Vector<value_type>  &src,
                      const std::pair<unsigned int,unsigned int>          &cell_range);

  void initialize_constraints(const std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator> > &periodic_face_pairs);

  FE_Q<dim>                  fe_wdist;
  DoFHandler<dim>  dof_handler_wdist;
  parallel::distributed::Vector<double> tauw_boundary;
  std::vector<unsigned int> vector_to_tauw_boundary;
  ConstraintMatrix constraint_periodic;
  std::vector<std::vector<LAPACKFullMatrix<value_type> > > matrices;
  FEParametersTauwN fe_param_n;
};

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesDualSplittingXWall<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
setup (const std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator> > periodic_face_pairs,
       std::set<types::boundary_id> dirichlet_bc_indicator,
       std::set<types::boundary_id> neumann_bc_indicator)
{
  this->setup(periodic_face_pairs,dirichlet_bc_indicator,neumann_bc_indicator);

  if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    std::cout << "\nXWall Initialization:" << std::endl;

  //initialize wall distance and closest wall-node connectivity
  if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    std::cout << "Initialize wall distance:...";
  init_wdist();
  if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    std::cout << " done!" << std::endl;

  //initialize some vectors
  this->data.initialize_dof_vector(this->fe_param.tauw, 2);
  this->fe_param.tauw = 1.0;
  fe_param_n.tauw =this->fe_param.tauw;

  matrices.resize(this->data.n_macro_cells());
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesDualSplittingXWall<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
create_dofs()
{

  // enumerate degrees of freedom
  // multigrid solvers for enrichment not supported
  this->dof_handler_u.distribute_dofs(*this->fe_u);
  this->dof_handler_p.distribute_dofs(this->fe_p);
  this->dof_handler_p.distribute_mg_dofs(this->fe_p);
  dof_handler_wdist.distribute_dofs(fe_wdist);

  unsigned int ndofs_per_cell_velocity    = Utilities::fixed_int_power<fe_degree+1,dim>::value*dim;
  unsigned int ndofs_per_cell_xwall    = Utilities::fixed_int_power<fe_degree_xwall+1,dim>::value*dim;
  unsigned int ndofs_per_cell_pressure    = Utilities::fixed_int_power<fe_degree_p+1,dim>::value;

  ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
  pcout << std::endl << "Discontinuous finite element discretization:" << std::endl << std::endl
    << "Velocity:" << std::endl
    << "  degree of 1D polynomials:\t"  << std::fixed << std::setw(10) << std::right << fe_degree
    << " (polynomial) and " << std::setw(10) << std::right << fe_degree_xwall << " (enrichment) " << std::endl
    << "  number of dofs per cell:\t"   << std::fixed << std::setw(10) << std::right << ndofs_per_cell_velocity
    << " (polynomial) and " << std::setw(10) << std::right << ndofs_per_cell_xwall << " (enrichment) " << std::endl
    << "  number of dofs (velocity):\t" << std::fixed << std::setw(10) << std::right << this->dof_handler_u.n_dofs() << std::endl
    << "Pressure:" << std::endl
    << "  degree of 1D polynomials:\t"  << std::fixed << std::setw(10) << std::right << fe_degree_p << std::endl
    << "  number of dofs per cell:\t"   << std::fixed << std::setw(10) << std::right << ndofs_per_cell_pressure << std::endl
    << "  number of dofs (pressure):\t" << std::fixed << std::setw(10) << std::right << this->dof_handler_p.n_dofs() << std::endl;
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesDualSplittingXWall<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
data_reinit(typename MatrixFree<dim,value_type>::AdditionalData & additional_data)
{
  std::vector<const DoFHandler<dim> * >  dof_handler_vec;

  dof_handler_vec.push_back(&this->dof_handler_u);
  dof_handler_vec.push_back(&this->dof_handler_p);
  dof_handler_vec.push_back(&dof_handler_wdist);

  std::vector<const ConstraintMatrix *> constraint_matrix_vec;
  ConstraintMatrix constraint, constraint_p;
  constraint.close();
  constraint_p.close();
  constraint_matrix_vec.push_back(&constraint);
  constraint_matrix_vec.push_back(&constraint_p);
  constraint_matrix_vec.push_back(&constraint_periodic);
  constraint_matrix_vec.push_back(&constraint);

  std::vector<Quadrature<1> > quadratures;

  // velocity
  quadratures.push_back(QGauss<1>(fe_degree+1));
  // pressure
  quadratures.push_back(QGauss<1>(fe_degree_p+1));
  // exact integration of nonlinear convective term
  quadratures.push_back(QGauss<1>(fe_degree + (fe_degree+2)/2));
  // enrichment
  quadratures.push_back(QGauss<1>(n_q_points_1d_xwall));

  this->data.reinit (this->mapping, dof_handler_vec, constraint_matrix_vec, quadratures, additional_data);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesDualSplittingXWall<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
push_back_constraint_matrix(std::vector<const ConstraintMatrix *> & constraint_matrix_vec)
{
  ConstraintMatrix constraint;
  constraint.close();
  constraint_matrix_vec.push_back(&constraint_periodic);
  constraint_matrix_vec.push_back(&constraint);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesDualSplittingXWall<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
init_wdist()
{
  // layout of aux_vector: 0-dim: normal, dim: distance, dim+1: nearest dof
  // index, dim+2: touch count (for computing weighted normals); normals not
  // currently used
  std::vector<parallel::distributed::Vector<double> > aux_vectors(dim+3);

  // store integer indices in a double. In order not to get overflow, we
  // need to make sure the global index fits into a double -> this limits
  // the maximum size in the dof indices to 2^53 (approx 10^15)
#ifdef DEAL_II_WITH_64BIT_INTEGERS
  AssertThrow(dof_handler_wdist.n_dofs() <
              (types::global_dof_index(1ull) << 53),
              ExcMessage("Sizes larger than 2^53 currently not supported"));
#endif

  IndexSet locally_relevant_set;
  DoFTools::extract_locally_relevant_dofs(this->dof_handler_wdist,
                                          locally_relevant_set);
  aux_vectors[0].reinit(this->dof_handler_wdist.locally_owned_dofs(),
                        locally_relevant_set, MPI_COMM_WORLD);
  for (unsigned int d=1; d<aux_vectors.size(); ++d)
    aux_vectors[d].reinit(aux_vectors[0]);

  // assign distance to close to infinity (we would like to use inf here but
  // there are checks in deal.II whether numbers are finite so we must use a
  // finite number here)
  const double unreached = 1e305;
  aux_vectors[dim] = unreached;

  // TODO: get the actual set of wall (Dirichlet) boundaries as input
  // arguments. Currently, this is matched with what is set in the outer
  // problem type.
  std::set<types::boundary_id> wall_boundaries;
  wall_boundaries.insert(0);

  // set the initial distance for the wall to zero and initialize the normal
  // directions
  {
    QGauss<dim-1> face_quadrature(1);
    FEFaceValues<dim> fe_face_values(this->fe_wdist, face_quadrature,
                                     update_normal_vectors);
    std::vector<types::global_dof_index> dof_indices(this->fe_wdist.dofs_per_face);
    int found = 0;
    for (typename DoFHandler<dim>::active_cell_iterator cell = this->dof_handler_wdist.begin_active(); cell != this->dof_handler_wdist.end(); ++cell)
      if (cell->is_locally_owned())
        for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
          if (cell->at_boundary(f) &&
              wall_boundaries.find(cell->face(f)->boundary_id()) !=
              wall_boundaries.end())
            {
              found = 1;
              cell->face(f)->get_dof_indices(dof_indices);
              // get normal vector on face
              fe_face_values.reinit(cell, f);
              const Tensor<1,dim> normal = fe_face_values.normal_vector(0);
              for (unsigned int i=0; i<dof_indices.size(); ++i)
                {
                  for (unsigned int d=0; d<dim; ++d)
                    aux_vectors[d](dof_indices[i]) += normal[d];
                  aux_vectors[dim](dof_indices[i]) = 0.;
                  if(constraint_periodic.is_constrained(dof_indices[i]))
                    aux_vectors[dim+1](dof_indices[i]) = (*constraint_periodic.get_constraint_entries(dof_indices[i]))[0].first;
                  else
                    aux_vectors[dim+1](dof_indices[i]) = dof_indices[i];
                  aux_vectors[dim+2](dof_indices[i]) += 1.;
                }
            }
    int found_global = Utilities::MPI::sum(found,MPI_COMM_WORLD);
    //at least one processor has to have walls
    AssertThrow(found_global>0, ExcMessage("Could not find any wall. Aborting."));
    for (unsigned int i=0; i<aux_vectors[0].local_size(); ++i)
      if (aux_vectors[dim+2].local_element(i) != 0)
        for (unsigned int d=0; d<dim; ++d)
          aux_vectors[d].local_element(i) /= aux_vectors[dim+2].local_element(i);
  }

  // this algorithm finds the closest point on the interface by simply
  // searching locally on each element. This algorithm is only correct for
  // simple meshes (as it searches purely locally and can result in zig-zag
  // paths that are nowhere near optimal on general meshes) but it works in
  // parallel when the mesh can be arbitrarily decomposed among
  // processors. A generic class of algorithms to find the closest point on
  // the wall (not necessarily on a node of the mesh) is by some interface
  // evolution method similar to finding signed distance functions to a
  // given interface (see e.g. Sethian, Level Set Methods and Fast Marching
  // Methods, 2000, Chapter 6). But I do not know how to keep track of the
  // point of origin in those algorithms which is essential here, so skip
  // that for the moment. -- MK, Dec 2015

  // loop as long as we have untracked degrees of freedom. this loop should
  // terminate after a number of steps that is approximately half the width
  // of the mesh in elements
  while (aux_vectors[dim].linfty_norm() == unreached)
    {
      aux_vectors[dim+2] = 0.;
      for (unsigned int d=0; d<dim+2; ++d)
        aux_vectors[d].update_ghost_values();

      // get a pristine vector with the content of the distances at the
      // beginning of the step to distinguish which degrees of freedom were
      // already touched before the current loop and which are in the
      // process of being updated
      parallel::distributed::Vector<double> distances_step(aux_vectors[dim]);
      distances_step.update_ghost_values();

      AssertThrow(this->fe_wdist.dofs_per_cell ==
                  GeometryInfo<dim>::vertices_per_cell, ExcNotImplemented());
      Quadrature<dim> quadrature(this->fe_wdist.get_unit_support_points());
      FEValues<dim> fe_values(this->fe_wdist, quadrature, update_quadrature_points);
      std::vector<types::global_dof_index> dof_indices(this->fe_wdist.dofs_per_cell);

      // go through all locally owned and ghosted cells and compute the
      // nearest point from within the element. Since we have both ghosted
      // and owned cells, we can be sure that the locally owned vector
      // elements get the closest point from the neighborhood
      for (typename DoFHandler<dim>::active_cell_iterator cell =
             this->dof_handler_wdist.begin_active();
           cell != this->dof_handler_wdist.end(); ++cell)
        if (!cell->is_artificial())
          {
            bool cell_is_initialized = false;
            cell->get_dof_indices(dof_indices);

            for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_cell; ++v)
              // point is unreached -> find the closest point within cell
              // that is already reached
              if (distances_step(dof_indices[v]) == unreached)
                {
                  for (unsigned int w=0; w<GeometryInfo<dim>::vertices_per_cell; ++w)
                    if (distances_step(dof_indices[w]) < unreached)
                      {
                        if (! cell_is_initialized)
                          {
                            fe_values.reinit(cell);
                            cell_is_initialized = true;
                          }

                        // here are the normal vectors in case they should
                        // be necessary in a refined version of the
                        // algorithm
                        /*
                        Tensor<1,dim> normal;
                        for (unsigned int d=0; d<dim; ++d)
                          normal[d] = aux_vectors[d](dof_indices[w]);
                        */
                        const Tensor<1,dim> distance_vec =
                          fe_values.quadrature_point(v) - fe_values.quadrature_point(w);
                        if (distances_step(dof_indices[w]) + distance_vec.norm() <
                            aux_vectors[dim](dof_indices[v]))
                          {
                            aux_vectors[dim](dof_indices[v]) =
                              distances_step(dof_indices[w]) + distance_vec.norm();
                            aux_vectors[dim+1](dof_indices[v]) =
                              aux_vectors[dim+1](dof_indices[w]);
                            for (unsigned int d=0; d<dim; ++d)
                              aux_vectors[d](dof_indices[v]) +=
                                aux_vectors[d](dof_indices[w]);
                            aux_vectors[dim+2](dof_indices[v]) += 1;
                          }
                      }
                }
          }
      for (unsigned int i=0; i<aux_vectors[0].local_size(); ++i)
        if (aux_vectors[dim+2].local_element(i) != 0)
          for (unsigned int d=0; d<dim; ++d)
            aux_vectors[d].local_element(i) /= aux_vectors[dim+2].local_element(i);
    }
  aux_vectors[dim+1].update_ghost_values();

  // at this point we could do a search for closer points in the
  // neighborhood of the points identified before (but it is probably quite
  // difficult to do and one needs to search in layers around a given point
  // to have all data available locally; I currently do not have a good idea
  // to sort out this mess and I am not sure whether we really need
  // something better than the local search above). -- MK, Dec 2015

  // copy the aux vector with extended ghosting into a vector that fits the
  // matrix-free partitioner
  this->data.initialize_dof_vector(this->fe_param.wdist, 2);
  AssertThrow(this->fe_param.wdist.local_size() == aux_vectors[dim].local_size(),
              ExcMessage("Vector sizes do not match, cannot import wall distances"));
  this->fe_param.wdist = aux_vectors[dim];
  this->fe_param.wdist.update_ghost_values();

  IndexSet accessed_indices(aux_vectors[dim+1].size());
  {
    // copy the accumulated indices into an index vector
    std::vector<types::global_dof_index> my_indices;
    my_indices.reserve(aux_vectors[dim+1].local_size());
    for (unsigned int i=0; i<aux_vectors[dim+1].local_size(); ++i)
      my_indices.push_back(static_cast<types::global_dof_index>(aux_vectors[dim+1].local_element(i)));
    // sort and compress out duplicates
    std::sort(my_indices.begin(), my_indices.end());
    my_indices.erase(std::unique(my_indices.begin(), my_indices.end()),
                     my_indices.end());
    accessed_indices.add_indices(my_indices.begin(),
                                 my_indices.end());
  }

  // create partitioner for exchange of ghost data (after having computed
  // the vector of wall shear stresses)
  std_cxx11::shared_ptr<const Utilities::MPI::Partitioner> vector_partitioner
    (new Utilities::MPI::Partitioner(this->dof_handler_wdist.locally_owned_dofs(),
                                     accessed_indices, MPI_COMM_WORLD));
  tauw_boundary.reinit(vector_partitioner);

  vector_to_tauw_boundary.resize(this->fe_param.wdist.local_size());
  for (unsigned int i=0; i<this->fe_param.wdist.local_size(); ++i)
    vector_to_tauw_boundary[i] = vector_partitioner->global_to_local
      (static_cast<types::global_dof_index>(aux_vectors[dim+1].local_element(i)));
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesDualSplittingXWall<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
precompute_inverse_mass_matrix()
{
  parallel::distributed::Vector<value_type> dummy;
  this->data.cell_loop(&DGNavierStokesDualSplittingXWall<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_precompute_mass_matrix,
                  this, dummy, dummy);
}

template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesDualSplittingXWall<dim,fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
local_precompute_mass_matrix (const MatrixFree<dim,value_type>        &data,
                              parallel::distributed::Vector<value_type>    &,
                              const parallel::distributed::Vector<value_type>  &,
                              const std::pair<unsigned int,unsigned int>   &cell_range)
{

  FEEval_Velocity_Velocity_linear fe_eval_velocity(data,this->fe_param,
      static_cast<typename std::underlying_type<DofHandlerSelector>::type >(DofHandlerSelector::velocity));
// FEEvaluationWrapper<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,1,value_type,true> fe_eval_xwall (data,this->param.xwallstatevec[0],xwallstatevec[1],0,3);

for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
{
  //first, check if we have an enriched element
  //if so, perform the routine for the enriched elements
  fe_eval_velocity.reinit (cell);

  AssertThrow((Utilities::fixed_int_power<fe_degree+1,dim>::value == fe_eval_velocity.dofs_per_cell),ExcMessage("wrong number of dofs"));
  if(fe_eval_velocity.enriched)
  {
    if(matrices[cell].size()==0)
      matrices[cell].resize(data.n_components_filled(cell));

    for (unsigned int v = 0; v < data.n_components_filled(cell); ++v)
    {
      if (matrices[cell][v].m() != fe_eval_velocity.dofs_per_cell)
        matrices[cell][v].reinit(fe_eval_velocity.dofs_per_cell, fe_eval_velocity.dofs_per_cell);// = onematrix;
      else
        matrices[cell][v]=0;
    }
    for (unsigned int j=0; j<fe_eval_velocity.dofs_per_cell; ++j)
    {
      for (unsigned int i=0; i<fe_eval_velocity.dofs_per_cell; ++i)
        fe_eval_velocity.write_cellwise_dof_value(i,make_vectorized_array(0.));
      fe_eval_velocity.write_cellwise_dof_value(j,make_vectorized_array(1.));

      fe_eval_velocity.evaluate (true,false,false);
      for (unsigned int q=0; q<fe_eval_velocity.n_q_points; ++q)
      {
//        std::cout << fe_eval_xwall.get_value(q)[0] << std::endl;
        fe_eval_velocity.submit_value (fe_eval_velocity.get_value(q), q);
      }
      fe_eval_velocity.integrate (true,false);

      for (unsigned int i=0; i<fe_eval_velocity.dofs_per_cell; ++i)
        for (unsigned int v = 0; v < data.n_components_filled(cell); ++v)
          if(fe_eval_velocity.component_enriched(v))
            (matrices[cell][v])(i,j) = (fe_eval_velocity.read_cellwise_dof_value(i))[v];
          else//this is a non-enriched element
          {
            if(i<fe_eval_velocity.std_dofs_per_cell && j<fe_eval_velocity.std_dofs_per_cell)
              (matrices[cell][v])(i,j) = (fe_eval_velocity.read_cellwise_dof_value(i))[v];
            else if(i == j)//diagonal
              (matrices[cell][v])(i,j) = 1.0;
          }
    }
//      for (unsigned int i=0; i<10; ++i)
//        std::cout << std::endl;
//      for (unsigned int v = 0; v < data.n_components_filled(cell); ++v)
//        matrix[v].print(std::cout,14,8);

    for (unsigned int v = 0; v < data.n_components_filled(cell); ++v)
    {
      (matrices[cell][v]).compute_lu_factorization();
    }
  }
}
//


}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesDualSplittingXWall<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
xwall_projection()
{

  for (unsigned int o=0; o < this->param.order; o++)
    this->data.cell_loop(&NavierStokesOperation<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_project_xwall,
                   this, this->velocity[o], this->velocity[o]);

}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesDualSplittingXWall<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
local_project_xwall (const MatrixFree<dim,value_type>        &data,
                     parallel::distributed::Vector<value_type>    &dst,
                     const parallel::distributed::Vector<value_type>  &src,
                     const std::pair<unsigned int,unsigned int>   &cell_range)
{

  FEEval_Velocity_Velocity_linear fe_eval_velocity_n(data,this->fe_param_n,
      static_cast<typename std::underlying_type<DofHandlerSelector>::type >(DofHandlerSelector::velocity));
//FEEvaluationXWall<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,1,value_type> fe_eval_xwall_n (data,xwallstatevec[0],*xwall.ReturnTauWN(),0,3);
FEEval_Velocity_Velocity_linear fe_eval_velocity(data,this->fe_param,
    static_cast<typename std::underlying_type<DofHandlerSelector>::type >(DofHandlerSelector::velocity));

for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
{
  //first, check if we have an enriched element
  //if so, perform the routine for the enriched elements
  fe_eval_velocity_n.reinit (cell);
  fe_eval_velocity.reinit (cell);
  if(fe_eval_velocity.enriched)
  {
    //now apply vectors to inverse matrix
    Vector<value_type> vector_result(fe_eval_velocity.dofs_per_cell);
    for (unsigned int idim = 0; idim < dim; ++idim)
    {
      fe_eval_velocity_n.read_dof_values(src.at(idim),src.at(idim+dim));
      fe_eval_velocity_n.evaluate(true,false);
      for (unsigned int q=0; q<fe_eval_velocity.n_q_points; q++)
        fe_eval_velocity.submit_value(fe_eval_velocity_n.get_value(q),q);
      fe_eval_velocity.integrate(true,false);
      for (unsigned int v = 0; v < this->data.n_components_filled(cell); ++v)
      {
        vector_result = 0;
        for (unsigned int i=0; i<fe_eval_velocity.dofs_per_cell; ++i)
          vector_result[i] = fe_eval_velocity.read_cellwise_dof_value(i)[v];
        (matrices[cell][v]).apply_lu_factorization(vector_result,false);
        for (unsigned int i=0; i<fe_eval_velocity.dofs_per_cell; ++i)
          fe_eval_velocity.write_cellwise_dof_value(i,vector_result[i],v);
      }
      fe_eval_velocity.set_dof_values (dst.at(idim),dst.at(idim+dim+1));
    }
  }
}
//


}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesDualSplittingXWall<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
update_tauw(parallel::distributed::Vector<value_type> &velocity)
{
  //store old wall shear stress
  fe_param_n.tauw.swap(this->fe_param.tauw);

  if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    std::cout << "\nCompute new tauw: ";
  CalculateWallShearStress(velocity,this->fe_param.tauw);
  //mean does not work currently because of all off-wall nodes in the vector
//    double tauwmean = tauw.mean_value();
//    std::cout << "mean = " << tauwmean << " ";

  value_type tauwmax = this->fe_param.tauw.linfty_norm();
  if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    std::cout << "max = " << tauwmax << " ";

  value_type minloc = 1e9;
  for(unsigned int i = 0; i < this->fe_param.tauw.local_size(); ++i)
  {
    if(this->fe_param.tauw.local_element(i)>0.0)
    {
      if(minloc > this->fe_param.tauw.local_element(i))
        minloc = this->fe_param.tauw.local_element(i);
    }
  }
  const value_type minglob = Utilities::MPI::min(minloc, MPI_COMM_WORLD);

  if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    std::cout << "min = " << minglob << " ";
  if(not this->param.variabletauw)
  {
    if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      std::cout << "(manually set to 1.0) ";
    this->fe_param.tauw = 1.0;
  }
  if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    std::cout << std::endl;
  this->fe_param.tauw.update_ghost_values();
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesDualSplittingXWall<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
calculate_wall_shear_stress (const parallel::distributed::Vector<value_type>   &src,
                                   parallel::distributed::Vector<value_type>      &dst)
{
  parallel::distributed::Vector<value_type> normalization;
  this->data.initialize_dof_vector(normalization, 2);
  parallel::distributed::Vector<value_type> force;
  this->data.initialize_dof_vector(force, 2);

  // initialize
  force = 0.0;
  normalization = 0.0;

  // run loop to compute the local integrals
  this->data.loop (&DGNavierStokesDualSplittingXWall<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_rhs_dummy,
      &DGNavierStokesDualSplittingXWall<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_rhs_dummy_face,
      &DGNavierStokesDualSplittingXWall<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_rhs_wss_boundary_face,
            this, force, src);

  this->data.loop (&DGNavierStokesDualSplittingXWall<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_rhs_dummy,
      &DGNavierStokesDualSplittingXWall<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_rhs_dummy_face,
      &DGNavierStokesDualSplittingXWall<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_rhs_normalization_boundary_face,
            this, normalization, src);

  // run normalization
  value_type mean = 0.0;
  unsigned int count = 0;
  for(unsigned int i = 0; i < force.local_size(); ++i)
  {
    if(normalization.local_element(i)>0.0)
    {
      tauw_boundary.local_element(i) = force.local_element(i) / normalization.local_element(i);
      mean += tauw_boundary.local_element(i);
      count++;
    }
  }
  mean = Utilities::MPI::sum(mean,MPI_COMM_WORLD);
  count = Utilities::MPI::sum(count,MPI_COMM_WORLD);
  mean /= (value_type)count;
  if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    std::cout << "mean = " << mean << " ";

  // communicate the boundary values for the shear stress to the calling
  // processor and access the data according to the vector_to_tauw_boundary
  // field
  tauw_boundary.update_ghost_values();

  for (unsigned int i=0; i<this->fe_param.tauw.local_size(); ++i)
    dst.local_element(i) = (1.-this->param.dtauw)*fe_param_n.tauw.local_element(i)+this->param.dtauw*tauw_boundary.local_element(vector_to_tauw_boundary[i]);
  dst.update_ghost_values();
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesDualSplittingXWall<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
local_rhs_dummy (const MatrixFree<dim,value_type>                &,
            parallel::distributed::Vector<value_type>      &,
            const parallel::distributed::Vector<value_type>  &,
            const std::pair<unsigned int,unsigned int>           &) const
{

}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesDualSplittingXWall<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
local_rhs_dummy_face (const MatrixFree<dim,value_type>                 &,
              parallel::distributed::Vector<value_type>      &,
              const parallel::distributed::Vector<value_type>  &,
              const std::pair<unsigned int,unsigned int>          &) const
{

}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesDualSplittingXWall<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
local_rhs_wss_boundary_face (const MatrixFree<dim,value_type>             &data,
                       parallel::distributed::Vector<value_type>    &dst,
                       const parallel::distributed::Vector<value_type>  &src,
                       const std::pair<unsigned int,unsigned int>          &face_range) const
{
//#ifdef XWALL
//  FEFaceEvaluationXWall<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,dim,value_type> fe_eval_xwall(data,wall_distance,tauw,true,0,3);
//  FEFaceEvaluation<dim,1,n_q_points_1d_xwall,1,value_type> fe_eval_tauw(data,true,2,3);
//#else
  FEFaceEval_Velocity_Velocity_linear fe_eval_velocity_face(data,this->fe_param,true,0);
//  FEFaceEvaluationXWall<dim,fe_degree,fe_degree_xwall,fe_degree+1,dim,value_type> fe_eval_xwall(data,wall_distance,tauw,true,0,0);
  FEFaceEvaluation<dim,1,fe_degree+1,1,value_type> fe_eval_tauw(data,true,2,0);

  for(unsigned int face=face_range.first; face<face_range.second; face++)
  {
    if (data.get_boundary_indicator(face) == 0) // Infow and wall boundaries
    {
      fe_eval_velocity_face.reinit (face);
      fe_eval_tauw.reinit (face);

      fe_eval_velocity_face.read_dof_values(src,0,src,dim+1);
      fe_eval_velocity_face.evaluate(false,true);
      if(fe_eval_velocity_face.n_q_points != fe_eval_tauw.n_q_points)
        std::cerr << "\nwrong number of quadrature points" << std::endl;

      for(unsigned int q=0;q<fe_eval_velocity_face.n_q_points;++q)
      {
        Tensor<1, dim, VectorizedArray<value_type> > average_gradient = fe_eval_velocity_face.get_normal_gradient(q);

        VectorizedArray<value_type> tauwsc = make_vectorized_array<value_type>(0.0);
        tauwsc = average_gradient.norm();
        tauwsc *= this->get_viscosity();
        fe_eval_tauw.submit_value(tauwsc,q);
      }
      fe_eval_tauw.integrate(true,false);
      fe_eval_tauw.distribute_local_to_global(dst);
    }
  }
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesDualSplittingXWall<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
local_rhs_normalization_boundary_face (const MatrixFree<dim,value_type>             &data,
                       parallel::distributed::Vector<value_type>    &dst,
                       const parallel::distributed::Vector<value_type>  &,
                       const std::pair<unsigned int,unsigned int>          &face_range) const
{
  FEFaceEvaluation<dim,1,fe_degree+1,1,value_type> fe_eval_tauw(data,true,2,0);
  for(unsigned int face=face_range.first; face<face_range.second; face++)
  {
    if (data.get_boundary_indicator(face) == 0) // Infow and wall boundaries
    {
      fe_eval_tauw.reinit (face);

      for(unsigned int q=0;q<fe_eval_tauw.n_q_points;++q)
        fe_eval_tauw.submit_value(make_vectorized_array<value_type>(1.0),q);

      fe_eval_tauw.integrate(true,false);
      fe_eval_tauw.distribute_local_to_global(dst);
    }
  }
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesDualSplittingXWall<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
initialize_constraints(const std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator> > &periodic_face_pairs)
{
  IndexSet xwall_relevant_set;
  DoFTools::extract_locally_relevant_dofs(this->dof_handler_wdist,
                                          xwall_relevant_set);
  constraint_periodic.clear();
  constraint_periodic.reinit(xwall_relevant_set);
  std::vector<GridTools::PeriodicFacePair<typename DoFHandler<dim>::cell_iterator> >
    periodic_face_pairs_dh(periodic_face_pairs.size());
  for (unsigned int i=0; i<periodic_face_pairs.size(); ++i)
    {
      GridTools::PeriodicFacePair<typename DoFHandler<dim>::cell_iterator> pair;
      pair.cell[0] = typename DoFHandler<dim>::cell_iterator
        (&periodic_face_pairs[i].cell[0]->get_triangulation(),
         periodic_face_pairs[i].cell[0]->level(),
         periodic_face_pairs[i].cell[0]->index(),
         &this->dof_handler_wdist);
      pair.cell[1] = typename DoFHandler<dim>::cell_iterator
        (&periodic_face_pairs[i].cell[1]->get_triangulation(),
         periodic_face_pairs[i].cell[1]->level(),
         periodic_face_pairs[i].cell[1]->index(),
         &this->dof_handler_wdist);
      pair.face_idx[0] = periodic_face_pairs[i].face_idx[0];
      pair.face_idx[1] = periodic_face_pairs[i].face_idx[1];
      pair.orientation = periodic_face_pairs[i].orientation;
      pair.matrix = periodic_face_pairs[i].matrix;
      periodic_face_pairs_dh[i] = pair;
    }
  DoFTools::make_periodicity_constraints<DoFHandler<dim> >(periodic_face_pairs_dh, constraint_periodic);
  DoFTools::make_hanging_node_constraints(this->dof_handler_wdist,
                                          constraint_periodic);

  constraint_periodic.close();
}

#endif /* INCLUDE_DGNAVIERSTOKESDUALSPLITTINGXWALL_H_ */