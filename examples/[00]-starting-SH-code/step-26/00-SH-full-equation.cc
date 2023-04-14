/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2013 - 2021 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------

 *
 * Author: Wolfgang Bangerth, Texas A&M University, 2013
 */


#include <deal.II/base/utilities.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/base/timer.h>

#include <fstream>
#include <iostream>
#include <random>


namespace Step26
{
  using namespace dealii;


  template <int dim, int spacedim>
  class HeatEquation
  {
  public:
    HeatEquation();
    HeatEquation(const unsigned int degree
                , double time_step_denominator
                , unsigned int ref_num
                , double r_constant = 0.5
                , double g1_constant = 0.5
                , double k_constant = 1.
                , double end_time = 0.5);
    void run();

    void test_matrices();

  private:
    void setup_system();
    void solve_time_step();
    void output_results() const;
    void refine_mesh(const unsigned int min_grid_level,
                     const unsigned int max_grid_level);


    const unsigned int degree;

    Triangulation<dim, spacedim> triangulation;
    FESystem<dim, spacedim>          fe;
    DoFHandler<dim, spacedim>    dof_handler;

    AffineConstraints<double> constraints;

    SparsityPattern      sparsity_pattern;

    //Not neeeded for current implementation
    /* SparseMatrix<double> mass_matrix;
    SparseMatrix<double> laplace_matrix; */
    SparseMatrix<double> system_matrix;

    Vector<double> solution;
    Vector<double> old_solution;
    Vector<double> system_rhs;

    double       time;
    double       time_step;
    unsigned int timestep_number;
    unsigned int timestep_denominator;
    unsigned int refinement_number;

    const double theta;
    const double r;
    const double g1;
    const double k;

    const double end_time;
  };




  template <int spacedim>
  class RightHandSide : public Function<spacedim>
  {
  public:
    RightHandSide()
      : Function<spacedim>()
      , period(0.2)
    {}

    virtual double value(const Point<spacedim> & p,
                         const unsigned int component = 0) const override;

  private:
    const double period;
  };



  template <int spacedim>
  double RightHandSide<spacedim>::value(const Point<spacedim> & p,
                                   const unsigned int component) const
  {
    (void)component;
    AssertIndexRange(component, 1);

    const double time = this->get_time();
    const double point_within_period =
      (time / period - std::floor(time / period));

    if ((point_within_period >= 0.0) && (point_within_period <= 0.2))
      {
        if ((p[0] > 0.5) && (p[1] > -0.5))
          return 1;
        else
          return 0;
      }
    else if ((point_within_period >= 0.5) && (point_within_period <= 0.7))
      {
        if ((p[0] > -0.5) && (p[1] > 0.5))
          return 1;
        else
          return 0;
      }
    else
      return 0;
  }



  template <int spacedim>
  class BoundaryValues : public Function<spacedim>
  {
  public:
    BoundaryValues()
      : Function<spacedim>(2)
    {}

    virtual double value(const Point<spacedim> & p,
                         const unsigned int component = 0) const override;
  };



  template <int spacedim>
  double BoundaryValues<spacedim>::value(const Point<spacedim> & /*p*/,
                                    const unsigned int component) const
  {
   (void)component;
    AssertIndexRange(component, 2);

    return 0.;

    if(component == 0){
      return 0;
    }
    else if (component == 1){
      return 0;
    }
    else{
      std::cout <<"An error has occured in [BoundaryValues]" << std::endl 
                << "Component out of range" << std::endl;
      return 0;
    }
  }

  template<int spacedim>
  class InitialConditions : public Function<spacedim>
  {
  private:
    const double r;
  public:
    InitialConditions()
      : Function<spacedim>(2),
        r(0.3)
    {}

    InitialConditions(const double r = 0.3)
      : Function<spacedim>(2),
        r(r)
    {}

    virtual double value(const Point<spacedim> &p,
                         const unsigned int component = 0) const override;
  };

  template<int spacedim>
  double InitialConditions<spacedim>::value(const Point<spacedim> &p,
                                     const unsigned int component) const
  {
    /*NOTE: This code now checks that we only have 2 FE components
            otherwise, we have not implemented the code correctly*/
    (void)component;
    AssertIndexRange(component, 2);

    // return 2.;

    if(component == 0){
     
      // return 2;
      // return std::rand()%200 - 100;

      // Random initial conditions
      
      std::random_device                          rand_dev;
      std::mt19937                                generator(rand_dev());
      std::uniform_real_distribution<double>      distro(-std::sqrt(r), std::sqrt(r));

      return distro(generator);

      if(p(0) > 0)
        return 1;
      else
        return -1;
    }
    else if (component == 1){
      return 1e18;
    }
    else{
      std::cout <<"An error has occured in [InitialtConditions]" << std::endl 
                << "Component out of range" << std::endl;
      return 0;
    }



    double rad_1 = std::pow((p(0) - 4), 2) + std::pow(p(1), 2) + std::pow(p(2), 2);
    double rad_2 = std::pow((p(0) + 4), 2) + std::pow(p(1), 2) + std::pow(p(2), 2);
    if(rad_1 < 1e-1){
      return 1.;
    }

    if(rad_2 < 1e-1){
      return -1.;
    }

    return 0;
  }
  
















































  template <int dim, int spacedim>
  HeatEquation<dim, spacedim>::HeatEquation()
    : degree(1)
    , fe(FE_Q<dim, spacedim>(degree), 2)
    , dof_handler(triangulation)
    , time_step(1. / 1500)
    , timestep_denominator(1500)
    , refinement_number(4)
    , theta(0.5)
    , r(0.5)
    , g1(0.5)
    , k(1.)
    , end_time(0.5)
  {}

  template <int dim, int spacedim>
  HeatEquation<dim, spacedim>::HeatEquation(const unsigned int degree,
                                            double       time_step_denominator,
                                            unsigned int ref_num,
                                            double       r_constant,
                                            double       g1_constant,
                                            double       k_constant,
                                            double       end_time)
    : degree(degree)
    , fe(FE_Q<dim, spacedim>(degree), 2)
    , dof_handler(triangulation)
    , time_step(1. / time_step_denominator)
    , timestep_denominator(time_step_denominator)
    , refinement_number(ref_num)
    , theta(0.5)
    , r(r_constant)
    , g1(g1_constant)
    , k(k_constant)
    , end_time(end_time)
  {}















































  template <int dim, int spacedim>
  void HeatEquation<dim, spacedim>::setup_system()
  {
    dof_handler.distribute_dofs(fe);

    const std::vector<types::global_dof_index> dofs_per_component =
      DoFTools::count_dofs_per_fe_component(dof_handler);
    const unsigned int n_u = dofs_per_component[0],
                       n_v = dofs_per_component[1];

    std::cout << "Number of active cells: " << triangulation.n_active_cells()
              << std::endl
              << "Total number of cells: " << triangulation.n_cells()
              << std::endl
              << "Number of degrees of freedom: " << dof_handler.n_dofs()
              << " (" << n_u << '+' << n_v << ')' << std::endl;

    DynamicSparsityPattern                dsp(dof_handler.n_dofs());

    DoFTools::make_sparsity_pattern(dof_handler,
                                    dsp);
    sparsity_pattern.copy_from(dsp);

    //Not used in current implementation
    /* mass_matrix.reinit(sparsity_pattern);
    laplace_matrix.reinit(sparsity_pattern); */
    system_matrix.reinit(sparsity_pattern);


    //Not sure if this works with an fe_system, so removing for now
    /* MatrixCreator::create_mass_matrix(dof_handler,
                                      QGauss<dim>(fe.degree + 1),
                                      mass_matrix);
    MatrixCreator::create_laplace_matrix(dof_handler,
                                         QGauss<dim>(fe.degree + 1),
                                         laplace_matrix); */

    solution.reinit(dof_handler.n_dofs());
    old_solution.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());
  }


  template <int dim, int spacedim>
  void HeatEquation<dim, spacedim>::solve_time_step()
  {
    /* SolverControl            solver_control(1000, 1e-8 * system_rhs.l2_norm());
    SolverCG<Vector<double>> cg(solver_control); */

    // std::cout << "Solving linear system" << std::endl;
    // Timer timer;


    SparseDirectUMFPACK direct_solver;

    direct_solver.initialize(system_matrix);

    direct_solver.vmult(solution, system_rhs);

    //Not currently using I think

    /* PreconditionSSOR<SparseMatrix<double>> preconditioner;
    preconditioner.initialize(system_matrix, 1.0);

    cg.solve(system_matrix, solution, system_rhs, preconditioner);

    constraints.distribute(solution); */

    // timer.stop();

    // std::cout << "done (" << timer.cpu_time() << " s)" << std::endl;
  }



  template <int dim, int spacedim>
  void HeatEquation<dim, spacedim>::output_results() const
  {
    std::vector<std::string> solution_names(1, "u");
    solution_names.emplace_back("v");
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      interpretation(1,
                     DataComponentInterpretation::component_is_scalar);
    interpretation.push_back(DataComponentInterpretation::component_is_scalar);

    DataOut<dim, spacedim> data_out;
    data_out.add_data_vector(dof_handler,
                             solution,
                             solution_names,
                             interpretation /*,
                             DataOut<dim, spacedim>::type_dof_data*/);

    data_out.build_patches(degree + 1);

    const std::string filename = 
      "solution-" + Utilities::int_to_string(timestep_number, 3) + ".vtu";

    std::ofstream output(filename);
    data_out.write_vtu(output);
  }



















































  template <int dim, int spacedim>
  void HeatEquation<dim, spacedim>::run()
  {
    // Vector<double> convergence_vector;

    GridGenerator::hyper_sphere(triangulation, Point<3>(0, 0, 0), 100);
    // GridGenerator::torus(triangulation, 3., 1.);
    // GridGenerator::hyper_cube(triangulation, -100, 100);
    triangulation.refine_global(refinement_number);

    setup_system();

    Vector<double> tmp;
    Vector<double> forcing_terms;

    time            = 0.0;
    timestep_number = 0;

    tmp.reinit(solution.size());
    forcing_terms.reinit(solution.size());

    InitialConditions<spacedim> initial_conditions(r);


    VectorTools::interpolate(dof_handler,
                             initial_conditions,
                             old_solution);
    solution = old_solution;

    // std::cout << old_solution << std::endl;

    output_results();

    //Attempting to construct the system matrix outside the loop,
    //so that I don't have to keep constructing it at each step.
    //
    //Note: Normally all the below variables are recreated at
    //each step, and I'm not sure if this will work if they
    //aren't, but I think it still should

    const QGauss<dim> quadrature_formula(degree + 2);

    FEValues<dim, spacedim> fe_values(fe, quadrature_formula, 
                                      update_values | update_gradients |
                                      update_quadrature_points | update_JxW_values);
    
    const unsigned int dofs_per_cell   = fe.n_dofs_per_cell();

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double> cell_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    const FEValuesExtractors::Scalar u(0);
    const FEValuesExtractors::Scalar v(1);

    for(const auto &cell : dof_handler.active_cell_iterators()){
      cell_matrix = 0;
      cell_rhs = 0;

      fe_values.reinit(cell);

      cell->get_dof_indices(local_dof_indices);

      for(const unsigned int q_index : fe_values.quadrature_point_indices()){       
        //NOTE: Going to try this, but we might run into a problem where u and v have differnt
        //      dof indices, in which case i will overfill my RHS vector, unless maybe the u
        //      finite elements are 0 in these cases, I'm not sure

        //NOTE: This does not happen, u and v are both a part of the same dof, at each node
        //      are 2 vector dof's, where the first component is u and the second is v.
        //      Typically if u has some value, then v is 0, and vice versa

        for(const unsigned int i : fe_values.dof_indices()){
          const double phi_i_u                   = fe_values[u].value(i, q_index);
          const Tensor<1, spacedim> grad_phi_i_u = fe_values[u].gradient(i, q_index);
          const double phi_i_v                   = fe_values[v].value(i, q_index);
          const Tensor<1, spacedim> grad_phi_i_v = fe_values[v].gradient(i, q_index);

          for(const unsigned int j : fe_values.dof_indices())
          {
            const double phi_j_u                   = fe_values[u].value(j, q_index);
            const Tensor<1, spacedim> grad_phi_j_u = fe_values[u].gradient(j, q_index);
            const double phi_j_v                   = fe_values[v].value(j, q_index);
            const Tensor<1, spacedim> grad_phi_j_v = fe_values[v].gradient(j, q_index);

            cell_matrix(i, j) += (phi_i_u*phi_j_u - time_step*r*phi_i_u*phi_j_u
                                    + time_step*phi_i_u*phi_j_v - time_step*grad_phi_i_u*grad_phi_j_v
                                    + phi_i_v*phi_j_u - grad_phi_i_v*grad_phi_j_u 
                                    - phi_i_v*phi_j_v)*fe_values.JxW(q_index);
          }
        }
      }

      for(unsigned int i : fe_values.dof_indices()){
        for(unsigned int j : fe_values.dof_indices()){
          system_matrix.add(local_dof_indices[i], 
                            local_dof_indices[j],
                            cell_matrix(i, j));
        }
      }
        }









    while (time <= end_time)
      {
        time += time_step;
        ++timestep_number;

        if(timestep_number%10 == 0){
          std::cout << "Time step " << timestep_number << " at t=" << time
                    << std::endl;
        }


        //Not currently using
        /* system_matrix.copy_from(mass_matrix);
        system_matrix.add(time_step, laplace_matrix);                   //Removing Laplace term for testing
        system_matrix.add((time_step - r*time_step), mass_matrix); */



        //Commented our a large section: Trying to define system_matrix once,
        //so all of this is defined above. Leaving for now in case it turns
        //out that I do need to redefine these at each step


        /* const QGauss<dim> quadrature_formula(degree + 2);

        FEValues<dim, spacedim> fe_values(fe, quadrature_formula, 
                                          update_values | update_gradients |
                                          update_quadrature_points | update_JxW_values);
        
        const unsigned int dofs_per_cell   = fe.n_dofs_per_cell();

        FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
        Vector<double> cell_rhs(dofs_per_cell);

        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

        const FEValuesExtractors::Scalar u(0);
        const FEValuesExtractors::Scalar v(1); */

        // Want this to be defined once above, so we shouldn't need to reset
        // at each step.

        // system_matrix = 0;


        system_rhs = 0;

        for(const auto &cell : dof_handler.active_cell_iterators()){
          cell_matrix = 0;
          cell_rhs = 0;

          fe_values.reinit(cell);

          cell->get_dof_indices(local_dof_indices);

          for(const unsigned int q_index : fe_values.quadrature_point_indices()){
            double Un1 = 0;

            // std::cout<<std::endl<<std::endl<<"In cell, outputing shape funtion values" << std::endl;

            for(const unsigned int i : fe_values.dof_indices()){
              // std::cout<<fe_values[u].value(i, q_index) << std::endl;
              Un1 += old_solution(local_dof_indices[i])*fe_values[u].value(i, q_index);
            }

            // std::cout<< "Un1: " << Un1 << std::endl;

            //NOTE: Going to try this, but we might run into a problem where u and v have differnt
            //      dof indices, in which case i will overfill my RHS vector, unless maybe the u
            //      finite elements are 0 in these cases, I'm not sure

            for(const unsigned int i : fe_values.dof_indices()){

              // Commented out a large section, work should be done above

              /* const double phi_i_u                   = fe_values[u].value(i, q_index);
              const Tensor<1, spacedim> grad_phi_i_u = fe_values[u].gradient(i, q_index);
              const double phi_i_v                   = fe_values[v].value(i, q_index);
              const Tensor<1, spacedim> grad_phi_i_v = fe_values[v].gradient(i, q_index);

              // std::cout << std::endl << "grad_phi_i_u: " << grad_phi_i_u << " grad_phi_i_v: " << grad_phi_i_v << std::endl;

              for(const unsigned int j : fe_values.dof_indices())
              {
                const double phi_j_u                   = fe_values[u].value(j, q_index);
                const Tensor<1, spacedim> grad_phi_j_u = fe_values[u].gradient(j, q_index);
                const double phi_j_v                   = fe_values[v].value(j, q_index);
                const Tensor<1, spacedim> grad_phi_j_v = fe_values[v].gradient(j, q_index);

                cell_matrix(i, j) += (phi_i_u*phi_j_u - time_step*r*phi_i_u*phi_j_u
                                        + time_step*phi_i_u*phi_j_v - time_step*grad_phi_i_u*grad_phi_j_v
                                        + phi_i_v*phi_j_u - grad_phi_i_v*grad_phi_j_u 
                                        - phi_i_v*phi_j_v)*fe_values.JxW(q_index);
                
                // std::cout<<cell_matrix(i, j)<<std::endl;
              } */

              


              cell_rhs(i) += (Un1 + time_step*g1*std::pow(Un1, 2) - time_step*k*std::pow(Un1, 3))
                              *fe_values[u].value(i, q_index)*fe_values.JxW(q_index);
            }
          }

          for(unsigned int i : fe_values.dof_indices()){
            //Commented our this section, work should be done above

            /* for(unsigned int j : fe_values.dof_indices()){
              system_matrix.add(local_dof_indices[i], 
                                local_dof_indices[j],
                                cell_matrix(i, j));
            } */

            system_rhs(local_dof_indices[i]) += cell_rhs(i);
          }


        }

        // std::cout << system_rhs << std::endl << std::endl;

        {
          BoundaryValues<spacedim> boundary_values_function;
          boundary_values_function.set_time(time);

          std::map<types::global_dof_index, double> boundary_values;
          VectorTools::interpolate_boundary_values(dof_handler,
                                                   0,
                                                   boundary_values_function,
                                                   boundary_values);

          MatrixTools::apply_boundary_values(boundary_values,
                                             system_matrix,
                                             solution,
                                             system_rhs);
        }

        // std::cout << system_rhs << std::endl << std::endl << std::endl << std::endl;

        /* std::cout << "System RHS" << std::endl << system_rhs << std::endl << std::endl;

        std::cout << "System Matrix" << std::endl;
        for(unsigned int i : fe_values.dof_indices()){
          for(unsigned int j : fe_values.dof_indices()){
            std::cout<< system_matrix(i, j) << " ";
          }
          std::cout << std::endl;
        } */

        solve_time_step();

        if(timestep_number%10 == 0){
          output_results();
        }



        // std::cout << std::endl << "GREP: "<< time << " " << solution[0] << std::endl;



        /* convergence_vector.reinit(solution.size());

        convergence_vector+= solution;
        convergence_vector-= old_solution;

        std::cout << "Solution difference has magnitude " << convergence_vector.l2_norm() << std::endl;

        if(convergence_vector.l2_norm() < 1e-6){
          std::cout << std::endl << std::endl;
          std::cout << "Solution converged at step " << timestep_number;
          std::cout << std::endl << std::endl;
          break;
        } */

        /* if(timestep_number >= 300){
          std::cout << std::endl << std::endl;
          std::cout << "Breaking loop, expected convergence" << std::endl;
          std::cout << std::endl << std::endl;
          break;
        } */

        old_solution = solution;
      }
  }
  
  
  
  
  template <int dim, int spacedim>
  void HeatEquation<dim, spacedim>::test_matrices()
  {
    Vector<double> tmp;

    GridGenerator::torus(triangulation, 3., 1.);
    // GridGenerator::hyper_cube(triangulation, -5, 5);
    triangulation.refine_global(refinement_number);

    InitialConditions<spacedim> initial_conditions;

    setup_system();

    tmp.reinit(solution.size());

    VectorTools::interpolate(dof_handler, initial_conditions, old_solution);

    //Not currently using mass and laplace matrices
    /* laplace_matrix.vmult(tmp, old_solution);

    std::cout << tmp << std::endl << std::endl;

    tmp.reinit(solution.size());

    mass_matrix.vmult(tmp, old_solution);

    std::cout << tmp << std::endl << std::endl; */

  }
} // namespace Step26


int main()
{
  try
    {
      using namespace Step26;

      HeatEquation<2, 3> heat_equation_solver(1, 100, 6, 0.3, 0.0, 1., 80.);
      heat_equation_solver.run();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  /* for(unsigned int i = 3; i < 7; ++i){
    for(unsigned int j = 1; j<21; ++j){
      std::cout << std::endl;
      std::cout << "=============================================================" << std::endl;
      std::cout << "Running step-26 with refinement " << i << " and timestep 1 over " << j*200 << std::endl << std::endl;
      {
        try
          {
            using namespace Step26;

            HeatEquation<2, 2> heat_equation_solver(j*200, i);
            heat_equation_solver.run();
          }
        catch (std::exception &exc)
          {
            std::cerr << std::endl
                      << std::endl
                      << "----------------------------------------------------"
                      << std::endl;
            std::cerr << "Exception on processing: " << std::endl
                      << exc.what() << std::endl
                      << "Aborting!" << std::endl
                      << "----------------------------------------------------"
                      << std::endl;

            return 1;
          }
        catch (...)
          {
            std::cerr << std::endl
                      << std::endl
                      << "----------------------------------------------------"
                      << std::endl;
            std::cerr << "Unknown exception!" << std::endl
                      << "Aborting!" << std::endl
                      << "----------------------------------------------------"
                      << std::endl;
            return 1;
          }
      }
    }
  }
 */
  return 0;
}
