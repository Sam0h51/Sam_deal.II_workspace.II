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

#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_tools.h>

#include <boost/math/special_functions/ellint_1.hpp>

#include <fstream>
#include <iostream>
#include <random>


namespace SwiftHohenbergSolver
{
  using namespace dealii;



  /// @brief This enum defines the four mesh types implemented
  ///        in this program and allows the user to pass which
  ///        mesh is desired to the solver at runtime. This is
  ///        useful for looping over different meshes.
  enum MeshType {HYPERCUBE, CYLINDER, SPHERE, TORUS, SINUSOID};


  enum InitialConditionType {HOTSPOT, PSUEDORANDOM, RANDOM};




  /// @brief This function warps points on a cyclindrical mesh by cosine wave along the central axis.
  ///        We use this function to generate the "sinusoid" mesh, which is the surface of revolution
  ///        bounded by the cosine wave.
  /// @tparam spacedim This is the dimension of the embedding space, which is where the input point lives
  /// @param p This is thel input point to be translated.
  /// @return The return as a tranlated point in the same dimensional space. This is the new point on the mesh.
  template<int spacedim>
  Point<spacedim> transform_function(const Point<spacedim>&p)
  {
    // Currently this only works for a 3-dimensional embedding space
    // because we are explicitly referencing the x, y, and z coordinates
    Assert(spacedim == 3, ExcNotImplemented());

    // Retruns a point where the x-coordinate is unchanged but the y and z coordinates are adjusted
    // by a cos wave of period 20, amplitude .5, and vertical shift 1
    return Point<spacedim>(p(0), p(1)*(1 + .5*std::cos((3.14159/10)*p(0))), p(2)*(1 + .5*std::cos((3.14159/10)*p(0))));
  }


  /// @brief Not currently implemented, but will function the same as above only with and undulary boundary curve rather
  ///        than a cosine boundary curve.
  /// @tparam spacedim See above
  /// @param p See above
  /// @return See above
  template<int spacedim>
  Point<spacedim> transform_function_2_electric_boogaloo(const Point<spacedim> &p)
  {
    Assert(spacedim == 3, ExcNotImplemented());
    return 0;
  }







  /// @brief  This is the class that holds all the important variables for the solver, as well as the important member
  ///         functions. This class is based off the HeatEquation class from step-26, so we won't go into full detail
  ///         on all the features, but we will highlight what has been changed for this problem.
  /// @tparam dim       This is the intrinsic dimension of the manifold we are solving on.
  /// @tparam spacedim  This is the dimension of the embedding space.
  template <int dim, int spacedim, MeshType MESH, InitialConditionType ICTYPE>
  class SHEquation
  {
  public:
    SHEquation();
    SHEquation(const unsigned int degree
                , double time_step_denominator
                , unsigned int ref_num
                /* , unsigned int iteration_number */
                , double r_constant = 0.5
                , double g1_constant = 0.5
                , std::string output_file_name = "solution-"
                , double end_time = 0.5);
    void run();

  private:
    void setup_system();
    void solve_time_step();
    void output_results() const;
    void make_grid();

    void make_cylinder();
    void make_sinusoid();
    void make_sphere();
    void make_torus();
    void make_hypercube();


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

    /* const unsigned int iteration_number; */

    const double theta;
    const double r;
    const double g1;
    const double k;

    const std::string output_file_name;

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
  double BoundaryValues<spacedim>::value(const Point<spacedim> & p,
                                    const unsigned int component) const
  {
   (void)component;
    AssertIndexRange(component, 2);

    return 0.;

    if(component == 0){
      if(p(0) < 0){
        return -1;
      }
      else{
        return 1;
      }
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

  template<int spacedim, MeshType MESH, InitialConditionType ICTYPE>
  class InitialCondition : public Function<spacedim>
  {
    private:
      const double r;
      Point<spacedim> center;
      double radius;
      double x_sin_coefficients[10];
      double y_sin_coefficients[10];

    public:
      InitialCondition()
      : Function<spacedim>(2),
        r(0.3),
        center(0),
        radius(.5)
      {
        for(int i = 0; i < 10; ++i){
          x_sin_coefficients[i] = 2*std::sqrt(r)*(std::rand()%1001)/1000 - std::sqrt(r);
          y_sin_coefficients[i] = 2*std::sqrt(r)*(std::rand()%1001)/1000 - std::sqrt(r);
        }
      }

      InitialCondition(const double r,
                        const double radius)
      : Function<spacedim>(2),
        r(r),
        radius(radius)
      {
        for(int i = 0; i < 10; ++i){
          x_sin_coefficients[i] = 2*std::sqrt(r)*(std::rand()%1001)/1000 - std::sqrt(r);
          y_sin_coefficients[i] = 2*std::sqrt(r)*(std::rand()%1001)/1000 - std::sqrt(r);
        }
      }

      virtual double value(const Point<spacedim> &p, const unsigned int component) const override;
  };

  template <>
  double InitialCondition<2, HYPERCUBE, HOTSPOT>::value(
    const Point<2> &p,
    const unsigned int     component) const
  {
    if(component == 0){
      if(p.square() <= radius){
        return std::sqrt(r);
      }
      else{
        return -std::sqrt(r);
      }
    }
    else{
      return 1e18;
    }
  }

  template <>
  double InitialCondition<3, CYLINDER, HOTSPOT>::value(
    const Point<3> &p,
    const unsigned int     component) const
  {
    if(component == 0){
      const Point<3> center(0, 0, 6);
      const Point<3> compare(p - center);
      if(compare.square() <= radius){
        return std::sqrt(r);
      }
      else{
        return -std::sqrt(r);
      }
    }
    else{
      return 1e18;
    }
  }

  template <>
  double InitialCondition<3, SPHERE, HOTSPOT>::value(
    const Point<3> &p,
    const unsigned int     component) const
  {
    if(component == 0){
      const Point<3> center(18.41988074, 0, 0);
      const Point<3> compare(p - center);
      if(compare.square() <= radius){
        return std::sqrt(r);
      }
      else{
        return -std::sqrt(r);
      }
    }
    else{
      return 1e18;
    }
  }

  template <>
  double InitialCondition<3, TORUS, HOTSPOT>::value(
    const Point<3> &p,
    const unsigned int     component) const
  {
    if(component == 0){
      const Point<3> center(13., 0, 0);
      const Point<3> compare(p - center);
      if(compare.square() <= radius){
        return std::sqrt(r);
      }
      else{
        return -std::sqrt(r);
      }
    }
    else{
      return 1e18;
    }
  }

  template <>
  double InitialCondition<3, SINUSOID, HOTSPOT>::value(
    const Point<3> &p,
    const unsigned int     component) const
  {
    if(component == 0){
      const Point<3> center(0, 0, 9.);
      const Point<3> compare(p - center);
      if(compare.square() <= radius){
        return std::sqrt(r);
      }
      else{
        return -std::sqrt(r);
      }
    }
    else{
      return 1e18;
    }
  }

  template <>
  double InitialCondition<2, HYPERCUBE, PSUEDORANDOM>::value(
    const Point<2> &p,
    const unsigned int     component) const
  {
    if(component == 0){
      double x_val = 0;
      double y_val = 0;
      for(int i=0; i < 10; ++i){
        x_val += x_sin_coefficients[i]*std::sin(2*3.141592653*p(0)/((i+1)*1.178097245));
        y_val += y_sin_coefficients[i]*std::sin(2*3.141592653*p(1)/((i+1)*1.178097245));
      }

      return x_val*y_val;
    }
    else{
      return 1e18;
    }
  }

  template <>
  double InitialCondition<3, CYLINDER, PSUEDORANDOM>::value(
    const Point<3> &p,
    const unsigned int     component) const
  {
    if(component == 0){
      double x_val = 0;
      double w_val = 0;
      double width = ((std::atan2(p(1),p(2)) - 3.1415926)/3.1415926)*18.84955592;
      for(int i=0; i < 10; ++i){
        x_val += x_sin_coefficients[i]*std::sin(2*3.141592653*p(0)/((i+1)*1.178097245));
        w_val += y_sin_coefficients[i]*std::sin(2*3.141592653*width/((i+1)*1.178097245));
      }

      return x_val*w_val;
    }
    else{
      return 1e18;
    }
  }

  template <>
  double InitialCondition<3, SPHERE, PSUEDORANDOM>::value(
    const Point<3> &p,
    const unsigned int     component) const
  {
    if(component == 0){
      double x_val = 0;
      double y_val = 0;
      for(int i=0; i < 10; ++i){
        x_val += x_sin_coefficients[i]*std::sin(2*3.141592653*p(0)/((i+1)*1.178097245));
        y_val += y_sin_coefficients[i]*std::sin(2*3.141592653*p(1)/((i+1)*1.178097245));
      }

      return x_val*y_val;
    }
    else{
      return 1e18;
    }
  }

  template <>
  double InitialCondition<3, TORUS, PSUEDORANDOM>::value(
    const Point<3> &p,
    const unsigned int     component) const
  {
    if(component == 0){
      double x_val = 0;
      double z_val = 0;
      for(int i=0; i < 10; ++i){
        x_val += x_sin_coefficients[i]*std::sin(2*3.141592653*p(0)/((i+1)*1.178097245));
        z_val += y_sin_coefficients[i]*std::sin(2*3.141592653*p(2)/((i+1)*1.178097245));
      }

      return x_val*z_val;
    }
    else{
      return 1e18;
    }
  }

  template <>
  double InitialCondition<3, SINUSOID, PSUEDORANDOM>::value(
    const Point<3> &p,
    const unsigned int     component) const
  {
    if(component == 0){
      double x_val = 0;
      double w_val = 0;
      double width = ((std::atan2(p(1),p(2)) - 3.1415926)/3.1415926)*18.84955592;
      for(int i=0; i < 10; ++i){
        x_val += x_sin_coefficients[i]*std::sin(2*3.141592653*p(0)/((i+1)*1.178097245));
        w_val += y_sin_coefficients[i]*std::sin(2*3.141592653*width/((i+1)*1.178097245));
      }

      return x_val*w_val;
    }
    else{
      return 1e18;
    }
  }

  template <>
  double InitialCondition<2, HYPERCUBE, RANDOM>::value(
    const Point<2> &/*p*/,
    const unsigned int     component) const
  {
    if(component == 0){
      return 2*std::sqrt(r)*(std::rand()%10001)/10000 - std::sqrt(r);
    }
    else{
      return 1e18;
    }
  }

  template <>
  double InitialCondition<3, CYLINDER, RANDOM>::value(
    const Point<3> &/*p*/,
    const unsigned int     component) const
  {
    if(component == 0){
      return 2*std::sqrt(r)*(std::rand()%10001)/10000 - std::sqrt(r);
    }
    else{
      return 1e18;
    }
  }

  template <>
  double InitialCondition<3, SPHERE, RANDOM>::value(
    const Point<3> &/*p*/,
    const unsigned int     component) const
  {
    if(component == 0){
      return 2*std::sqrt(r)*(std::rand()%10001)/10000 - std::sqrt(r);
    }
    else{
      return 1e18;
    }
  }

  template <>
  double InitialCondition<3, TORUS, RANDOM>::value(
    const Point<3> &/*p*/,
    const unsigned int     component) const
  {
    if(component == 0){
      return 2*std::sqrt(r)*(std::rand()%10001)/10000 - std::sqrt(r);
    }
    else{
      return 1e18;
    }
  }

  template <>
  double InitialCondition<3, SINUSOID, RANDOM>::value(
    const Point<3> &/*p*/,
    const unsigned int     component) const
  {
    if(component == 0){
      return 2*std::sqrt(r)*(std::rand()%10001)/10000 - std::sqrt(r);
    }
    else{
      return 1e18;
    }
  }


  /* template<int spacedim>
  class InitialCondition : public Function<spacedim>
  {
  private:
    const double r;
    double x_sin_coefficients[10];
    double y_sin_coefficients[10];
  public:
    InitialCondition()
      : Function<spacedim>(2),
        r(0.3)
    {
      for(int i = 0; i < 10; ++i){
        x_sin_coefficients[i] = 2*std::sqrt(r)*(std::rand()%1001)/1000 - std::sqrt(r);
        y_sin_coefficients[i] = 2*std::sqrt(r)*(std::rand()%1001)/1000 - std::sqrt(r);
      }
    }

    InitialCondition(const double r = 0.3)
      : Function<spacedim>(2),
        r(r)
    {
      for(int i = 0; i < 10; ++i){
        x_sin_coefficients[i] = 2*std::sqrt(r)*(std::rand()%10001)/10000 - std::sqrt(r);
        y_sin_coefficients[i] = 2*std::sqrt(r)*(std::rand()%10001)/10000 - std::sqrt(r);
      }
    }

    virtual double value(const Point<spacedim> &p,
                         const unsigned int component = 0) const override;
  };
 */
  
  /* template<int spacedim>
  double InitialCondition<spacedim>::value(const Point<spacedim> &p,
                                     const unsigned int component) const
  {
    //NOTE: This code now checks that we only have 2 FE components
    //      otherwise, we have not implemented the code correctly
    (void)component;
    AssertIndexRange(component, 2);

    // return std::atan(p(0));

    // return 2.;

    if(component == 0){
     
      // return 2;
      // return std::rand()%200 - 100;

      // Random initial conditions

      // const unsigned int fixed_seed = 314159;

      // return 2*std::sqrt(3)*(std::rand()%10001)/10000 - std::sqrt(3);
      
      std::random_device                          rand_dev;
      std::mt19937                                generator(fixed_seed);
      std::uniform_real_distribution<double>      distro(-std::sqrt(r), std::sqrt(r));

      return distro(generator);

      double x_val = 0;
      double y_val = 0;

      for(int i=0; i < 10; ++i){
        x_val += x_sin_coefficients[i]*std::sin(2*3.141592653*p(0)/((i+1)*1.178097245));
        y_val += y_sin_coefficients[i]*std::sin(2*3.141592653*p(1)/((i+1)*1.178097245));
      }

      return x_val*y_val;

      Assert(spacedim == 2, ExcNotImplemented());

      Point<spacedim> center(0, 1);

      Point<spacedim> compare(p - center);

      if(compare.square() < .5){
        return 3.;
      }
      else{
        return -3;
      }


      // if(p(0) > 0)
      //   return 1;
      // else
      //   return -1;
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
  
 */

  namespace InitialConditions{
    ////////////////////////
    // Class Definitions: //
    ////////////////////////

    template<int spacedim>
    class Random : public Function<spacedim>
    {
      private:
        const double r;
      public:
        Random()
          : Function<spacedim>(2),
            r(0.3)
        {}
        Random(const double r = 0.3)
          : Function<spacedim>(2),
            r(r)
        {}

        virtual double value(const Point<spacedim> &p, const unsigned int component) const override;
    };

    template<int spacedim>
    class ArcTan_x : public Function<spacedim>
    {
      public:
        ArcTan_x()
          : Function<spacedim>(2)
        {}

        virtual double value(const Point<spacedim> &p,
                              const unsigned int     component) const override;
    };

    template<int spacedim>
    class ArcTan_y : public Function<spacedim>
    {
      public:
        ArcTan_y()
          : Function<spacedim>(2)
        {}

        virtual double value(const Point<spacedim> &p,
                              const unsigned int    component) const override;
    };

    /* template<int spacedim>
    class Psuedorandom_HYPERCUBE : public Function<spacedim>
    {
    private:
      const double r;
      double x_sin_coefficients[10];
      double y_sin_coefficients[10];
    public:
      Psuedorandom_HYPERCUBE()
        : Function<spacedim>(2),
          r(0.3)
      {
        for(int i = 0; i < 10; ++i){
          x_sin_coefficients[i] = 2*std::sqrt(r)*(std::rand()%1001)/1000 - std::sqrt(r);
          y_sin_coefficients[i] = 2*std::sqrt(r)*(std::rand()%1001)/1000 - std::sqrt(r);
        }
      }

      Psuedorandom_HYPERCUBE(const double r = 0.3)
        : Function<spacedim>(2),
          r(r)
      {
        for(int i = 0; i < 10; ++i){
          x_sin_coefficients[i] = 2*std::sqrt(r)*(std::rand()%10001)/10000 - std::sqrt(r);
          y_sin_coefficients[i] = 2*std::sqrt(r)*(std::rand()%10001)/10000 - std::sqrt(r);
        }
      }

      virtual double value(const Point<spacedim> &p,
                          const unsigned int component = 0) const override;
    };

    template<int spacedim>
    class Psuedorandom_CYLINDER_and_SINUSOID : public Function<spacedim>
    {
    private:
      const double r;
      double x_sin_coefficients[10];
      double y_sin_coefficients[10];
    public:
      Psuedorandom_CYLINDER_and_SINUSOID()
        : Function<spacedim>(2),
          r(0.3)
      {
        for(int i = 0; i < 10; ++i){
          x_sin_coefficients[i] = 2*std::sqrt(r)*(std::rand()%1001)/1000 - std::sqrt(r);
          y_sin_coefficients[i] = 2*std::sqrt(r)*(std::rand()%1001)/1000 - std::sqrt(r);
        }
      }

      Psuedorandom_CYLINDER_and_SINUSOID(const double r = 0.3)
        : Function<spacedim>(2),
          r(r)
      {
        for(int i = 0; i < 10; ++i){
          x_sin_coefficients[i] = 2*std::sqrt(r)*(std::rand()%10001)/10000 - std::sqrt(r);
          y_sin_coefficients[i] = 2*std::sqrt(r)*(std::rand()%10001)/10000 - std::sqrt(r);
        }
      }

      virtual double value(const Point<spacedim> &p,
                          const unsigned int component = 0) const override;
    }; */

    template<int spacedim>
    class HotSpot : public Function<spacedim>
    {
      private:
        Point<spacedim> center;
        double radius;
      
      public:
        HotSpot(const Point<spacedim> center, const double radius = 0.5)
        : Function<spacedim>(2),
          center(center),
          radius(radius)
        {}

        virtual double value(const Point<spacedim> &p, const unsigned int component) const override;
    };

    template<int spacedim>
    class Psuedorandom : public Function<spacedim>
    {
    private:
      const double r;
      const MeshType MESH;
      double x_sin_coefficients[10];
      double y_sin_coefficients[10];
    public:
      Psuedorandom()
        : Function<spacedim>(2),
          r(0.3),
          MESH(HYPERCUBE)
      {
        for(int i = 0; i < 10; ++i){
          x_sin_coefficients[i] = 2*std::sqrt(r)*(std::rand()%1001)/1000 - std::sqrt(r);
          y_sin_coefficients[i] = 2*std::sqrt(r)*(std::rand()%1001)/1000 - std::sqrt(r);
        }
      }

      Psuedorandom(const MeshType mesh, const double r = 0.3)
        : Function<spacedim>(2),
          r(r),
          MESH(mesh)
      {
        for(int i = 0; i < 10; ++i){
          x_sin_coefficients[i] = 2*std::sqrt(r)*(std::rand()%10001)/10000 - std::sqrt(r);
          y_sin_coefficients[i] = 2*std::sqrt(r)*(std::rand()%10001)/10000 - std::sqrt(r);
        }
      }

      virtual double value(const Point<spacedim> &p,
                          const unsigned int component = 0) const override;
    };

    ///////////////////////////
    // Function Definitions: //
    ///////////////////////////

    template <int spacedim>
    double Random<spacedim>::value(const Point<spacedim> &p,
                                    const unsigned int     component) const
    {
      if(component==0){
        return 2*std::sqrt(r)*(std::rand()%10001)/10000 - std::sqrt(r);
      }
      else{
        return 1e18;
      }
    }

    template <int spacedim>
    double ArcTan_x<spacedim>::value(const Point<spacedim> &p,
                                      const unsigned int     component) const
    {
      if(component == 0){
        return std::atan(p(0));
      }
      else{
        return 1e18;
      }
    }
    template <int spacedim>
    double ArcTan_y<spacedim>::value(const Point <spacedim> & p,
                                      const unsigned int component) const
    {
      if(component==0){
        return std::atan(p(1));
      }
      else{
        return 1e18;
      }
    }
    

    /* template<int spacedim>
    double Psuedorandom_HYPERCUBE<spacedim>::value(const Point<spacedim> &p, const unsigned int component) const
    {
      if(component==0){
        double x_val = 0;
        double y_val = 0;

        for(int i=0; i < 10; ++i){
          x_val += x_sin_coefficients[i]*std::sin(2*3.141592653*p(0)/((i+1)*1.178097245));
          y_val += y_sin_coefficients[i]*std::sin(2*3.141592653*p(1)/((i+1)*1.178097245));
        }

        return x_val*y_val;
      }
      else{
        return 1e18;
      }
    }

    template <int spacedim>
    double
    Psuedorandom_CYLINDER_and_SINUSOID<spacedim>::value(const Point<spacedim> &p,
                                           const unsigned int component) const
    {
      if(component==0){
        double width = ((std::atan2(p(1)/p(2)) - 3.1415926)/3.1415926)*18.84955592;

        double x_val = 0;
        double w_val = 0;

        for(int i=0; i < 10; ++i){
          x_val += x_sin_coefficients[i]*std::sin(2*3.141592653*p(0)/((i+1)*1.178097245));
          x_val += y_sin_coefficients[i]*std::sin(2*3.141592653*p(1)/((i+1)*1.178097245));
        }

        return x_val*w_val;

      }
    } */


    template <int spacedim>
    double HotSpot<spacedim>::value(const Point<spacedim> &p,
                                    const unsigned int     component) const
    {
      if(component==0){
        Point<spacedim> difference(p - center);
        if(difference.square() <= radius){
          return 1.;
        }
        else{
          return -1.;
        }
      }
      else{
        return 1e18;
      }
    }

    template <int spacedim>
    double
    Psuedorandom<spacedim>::value(const Point<spacedim> &p,
                                        const unsigned int     component) const
    {
      if(component == 0){
        double x_val = 0;
        double y_val = 0;
        double w_val = 0;
        double width = 0;

        switch (MESH)
        {
        case HYPERCUBE:
          for(int i=0; i < 10; ++i){
            x_val += x_sin_coefficients[i]*std::sin(2*3.141592653*p(0)/((i+1)*1.178097245));
            y_val += y_sin_coefficients[i]*std::sin(2*3.141592653*p(1)/((i+1)*1.178097245));
          }

          return x_val*y_val;
          break;

        case CYLINDER:
          width = ((std::atan2(p(1),p(2)) - 3.1415926)/3.1415926)*18.84955592;

          for(int i=0; i < 10; ++i){
            x_val += x_sin_coefficients[i]*std::sin(2*3.141592653*p(0)/((i+1)*1.178097245));
            w_val += y_sin_coefficients[i]*std::sin(2*3.141592653*width/((i+1)*1.178097245));
          }

          return x_val*w_val;
        
        case SINUSOID:
          width = ((std::atan2(p(1),p(2)) - 3.1415926)/3.1415926)*18.84955592;

          for(int i=0; i < 10; ++i){
            x_val += x_sin_coefficients[i]*std::sin(2*3.141592653*p(0)/((i+1)*1.178097245));
            w_val += y_sin_coefficients[i]*std::sin(2*3.141592653*width/((i+1)*1.178097245));
          }

          return x_val*w_val;
        
        default:
          std::cout << "Psuedorandom called with MeshType != HYPERPLANE, CYLINDER, or SINUSOID. This is not implemented" << std::endl;
          return 0;
          break;
        }
      }
      else{
        return 1e18;
      }
    }
  } // namespace InitialConditions



  template <int dim, int spacedim, MeshType MESH, InitialConditionType ICTYPE>
  SHEquation<dim, spacedim, MESH, ICTYPE>::SHEquation()
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
    , output_file_name("solution-")
    , end_time(0.5)
    /* , iteration_number(0) */
  {}

  template <int dim, int spacedim, MeshType MESH, InitialConditionType ICTYPE>
  SHEquation<dim, spacedim, MESH, ICTYPE>::SHEquation(const unsigned int degree,
                                            double       time_step_denominator,
                                            unsigned int ref_num,
                                            /* unsigned int iteration_number, */
                                            double       r_constant,
                                            double       g1_constant,
                                            std::string  output_file_name,
                                            double       end_time)
    : degree(degree)
    , fe(FE_Q<dim, spacedim>(degree), 2)
    , dof_handler(triangulation)
    , time_step(1. / time_step_denominator)
    , timestep_denominator(time_step_denominator)
    , refinement_number(ref_num)
    /* , iteration_number(iteration_number) */
    , theta(0.5)
    , r(r_constant)
    , g1(g1_constant)
    , k(1.)
    , output_file_name(output_file_name)
    , end_time(end_time)
  {}















































  template <int dim, int spacedim, MeshType MESH, InitialConditionType ICTYPE>
  void SHEquation<dim, spacedim, MESH, ICTYPE>::setup_system()
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


  template <int dim, int spacedim, MeshType MESH, InitialConditionType ICTYPE>
  void SHEquation<dim, spacedim, MESH, ICTYPE>::solve_time_step()
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



  template <int dim, int spacedim, MeshType MESH, InitialConditionType ICTYPE>
  void SHEquation<dim, spacedim, MESH, ICTYPE>::output_results() const
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
      output_file_name + Utilities::int_to_string(timestep_number, 3) + ".vtu";

    std::ofstream output(filename);
    data_out.write_vtu(output);
  }

  template <>
  void SHEquation<2, 2, HYPERCUBE, HOTSPOT>::make_grid()
  {
    make_hypercube();
  }

  template <>
  void SHEquation<2, 3, CYLINDER, HOTSPOT>::make_grid()
  {
    make_cylinder();
  }

  template <>
  void SHEquation<2, 3, SPHERE, HOTSPOT>::make_grid()
  {
    make_sphere();
  }

  template <>
  void SHEquation<2, 3, TORUS, HOTSPOT>::make_grid()
  {
    make_torus();
  }

  template <>
  void SHEquation<2, 3, SINUSOID, HOTSPOT>::make_grid()
  {
    make_sinusoid();
  }

  template <>
  void SHEquation<2, 2, HYPERCUBE, PSUEDORANDOM>::make_grid()
  {
    make_hypercube();
  }

  template <>
  void SHEquation<2, 3, CYLINDER, PSUEDORANDOM>::make_grid()
  {
    make_cylinder();
  }

  template <>
  void SHEquation<2, 3, SPHERE, PSUEDORANDOM>::make_grid()
  {
    make_sphere();
  }

  template <>
  void SHEquation<2, 3, TORUS, PSUEDORANDOM>::make_grid()
  {
    make_torus();
  }

  template <>
  void SHEquation<2, 3, SINUSOID, PSUEDORANDOM>::make_grid()
  {
    make_sinusoid();
  }

  template <>
  void SHEquation<2, 2, HYPERCUBE, RANDOM>::make_grid()
  {
    make_hypercube();
  }

  template <>
  void SHEquation<2, 3, CYLINDER, RANDOM>::make_grid()
  {
    make_cylinder();
  }

  template <>
  void SHEquation<2, 3, SPHERE, RANDOM>::make_grid()
  {
    make_sphere();
  }

  template <>
  void SHEquation<2, 3, TORUS, RANDOM>::make_grid()
  {
    make_torus();
  }

  template <>
  void SHEquation<2, 3, SINUSOID, RANDOM>::make_grid()
  {
    make_sinusoid();
  }
















































  template <int dim, int spacedim, MeshType MESH, InitialConditionType ICTYPE>
  void SHEquation<dim, spacedim, MESH, ICTYPE>::run()
  {
    make_grid();
    /* if constexpr (MESH == HYPERCUBE){
      GridGenerator::hyper_cube(triangulation, -18.84955592, 18.84955592);
      triangulation.refine_global(refinement_number);
    }
    else if constexpr (MESH == CYLINDER){
      make_cylinder();
    }

    else if constexpr (MESH==SPHERE){
      make_sphere();
    }

    else if constexpr (MESH==TORUS){
      GridGenerator::torus(triangulation, 9., 4.);
      triangulation.refine_global(refinement_number - 2);
    }
    else if constexpr (MESH==SINUSOID)
      make_sinusoid(); */






    /* switch (mesh_type)
    {
    case HYPERCUBE:
    {
      GridGenerator::hyper_cube(triangulation, -18.84955592, 18.84955592);
      triangulation.refine_global(refinement_number);
      break;
    }
    case CYLINDER:
      {
      Triangulation<spacedim> cylinder;
      GridGenerator::cylinder(cylinder, 6, 18.84955592);

      GridGenerator::extract_boundary_mesh(cylinder, triangulation, {0});

      const CylindricalManifold<dim, spacedim> boundary;
      triangulation.set_all_manifold_ids(0);
      triangulation.set_manifold(0, boundary);

      triangulation.refine_global(refinement_number);
      }
      break;
    case SPHERE:
      GridGenerator::hyper_sphere(triangulation, Point<3>(0, 0, 0), 18.41988074);
      triangulation.refine_global(refinement_number);
      break;
    case TORUS:
      GridGenerator::torus(triangulation, 9., 4.);
      triangulation.refine_global(refinement_number);
      break;
    case SINUSOID:
      {
      Triangulation<spacedim> cylinder;
      GridGenerator::cylinder(cylinder, 6, 18.84955592);

      GridGenerator::extract_boundary_mesh(cylinder, triangulation, {0});

      const CylindricalManifold<dim, spacedim> boundary;
      triangulation.set_all_manifold_ids(0);
      triangulation.set_manifold(0, boundary);

      triangulation.refine_global(refinement_number);

      GridTools::transform(transform_function<spacedim>, triangulation);
      }
      break;
    default:
      std::cout << "No mesh specified, using HYPERCUBE(-1, 1)" << std::endl;
      GridGenerator::hyper_cube(triangulation, -1, 1);
      br eak;
    }*/


    //Setup for an unduloid mesh
    /* Triangulation<spacedim> cylinder;
    GridGenerator::cylinder(cylinder, 10, 25);

    GridGenerator::extract_boundary_mesh(cylinder, triangulation, {0});

    const CylindricalManifold<dim, spacedim> boundary;
    triangulation.set_all_manifold_ids(0);
    triangulation.set_manifold(0, boundary); */

    // triangulation.refine_global(refinement_number);

    // GridTools::transform(transform_function<spacedim>, triangulation);

    setup_system();

    Vector<double> tmp;
    Vector<double> forcing_terms;

    time            = 0.0;
    timestep_number = 0;

    tmp.reinit(solution.size());
    forcing_terms.reinit(solution.size());

    /* if(initial_condition_type == HOTSPOT){
      Point<spacedim> hotspot_center;
      switch (mesh_type)
      {
      case HYPERCUBE:
        if(spacedim==2){
          {
            Point<spacedim> center(0, 0);
            hotspot_center = center;
          }
        }
        else if(spacedim==3){
          {
            Point<spacedim> center(0, 0, 0);
            hotspot_center = center;
          }
        }
        else{
          std::cout << "Trying to use HYPERCUBE with spacedim != 2, 3: This is not implemented" << std::endl;
          throw ExcNotImplemented();
        }
        break;
      case CYLINDER:
        {
          Point<spacedim> center(0, 6, 0);
          hotspot_center = center;
        }
        break;
      case SPHERE:
        {
          Point<spacedim> center(18.41988074, 0, 0);
          hotspot_center = center;
        }
        break;
      case TORUS:
        {
          Point<spacedim> center(0, 13., 0);
          hotspot_center = center;
        }
        break;
      case SINUSOID:
        {
          Point<spacedim> center(0, 6, 0);
          hotspot_center = center;
        }
        break;
      default:
        std::cout << "Could not determine HotSpot center from mesh type, defaulting to origin" << std::endl;
        if(spacedim == 3){
          {
            Point<spacedim> center(0, 0, 0);
            hotspot_center = center;
          }
        }
        else if(spacedim == 2){
          {
            Point<spacedim> center(0, 0);
            hotspot_center = center;
          }
        }
        else{
          std::cout << "spacedim wasn't 2 or 3, default HotSpot center not implemented" <<std::endl;
          throw ExcNotImplemented();
        }
        break;
      }
      InitialConditions::HotSpot<spacedim> initial_conditions(hotspot_center);
    }
    else if(initial_condition_type == ARCTAN_X){
      InitialConditions::ArcTan_x<spacedim> initial_conditions();
    }
    else if(initial_condition_type == ARCTAN_Y){
      InitialConditions::ArcTan_y<spacedim> initial_conditions();
    }
    else if(initial_condition_type == PSUEDORANDOM){
      std::srand(314);
      if(mesh_type == HYPERCUBE){
        InitialConditions::Psuedorandom_HYPERCUBE<spacedim> initial_conditions();
      }
      else if(mesh_type == CYLINDER){
        InitialConditions::Psuedorandom_CYLINDER_and_SINUSOID<spacedim> initial_conditions();
      }
      else{
        InitialConditions::Psuedorandom_CYLINDER_and_SINUSOID<spacedim> initial_conditions();
      }
    }
    else if(initial_condition_type == FIXED_RANDOM){
      std::srand(314);
      InitialConditions::Random<spacedim> initial_conditions();
    }
    else{
      InitialConditions::Random<spacedim> initial_conditions();
    }*/

    /* switch (initial_condition_type)
      {
      case HOTSPOT:
        Point<spacedim> hotspot_center;
        switch (mesh_type)
        {
        case HYPERCUBE:
          if(spacedim==2){
            {
              Point<spacedim> center(0, 0);
              hotspot_center = center;
            }
          }
          else if(spacedim==3){
            {
              Point<spacedim> center(0, 0, 0);
              hotspot_center = center;
            }
          }
          else{
            std::cout << "Trying to use HYPERCUBE with spacedim != 2, 3: This is not implemented" << std::endl;
            throw ExcNotImplemented();
          }
          break;
        case CYLINDER:
          {
            Point<spacedim> center(0, 6, 0);
            hotspot_center = center;
          }
          break;
        case SPHERE:
          {
            Point<spacedim> center(18.41988074, 0, 0);
            hotspot_center = center;
          }
          break;
        case TORUS:
          {
            Point<spacedim> center(0, 13., 0);
            hotspot_center = center;
          }
          break;
        case SINUSOID:
          {
            Point<spacedim> center(0, 9, 0);
            hotspot_center = center;
          }
          break;
        default:
          std::cout << "Could not determine HotSpot center from mesh type, defaulting to origin" << std::endl;
          if(spacedim == 3){
            {
              Point<spacedim> center(0, 0, 0);
              hotspot_center = center;
            }
          }
          else if(spacedim == 2){
            {
              Point<spacedim> center(0, 0);
              hotspot_center = center;
            }
          }
          else{
            std::cout << "spacedim wasn't 2 or 3, default HotSpot center not implemented" <<std::endl;
            throw ExcNotImplemented();
          }
          break;
        }
        InitialConditions::HotSpot<spacedim> initial_conditions(hotspot_center);
        break;

      case ARCTAN_X:
        InitialConditions::ArcTan_x<spacedim> initial_conditions();
        break;

      case ARCTAN_Y:
        InitialConditions::ArcTan_y<spacedim> initial_conditions();
        break;

      case PSUEDORANDOM:
        Assert(mesh_type == HYPERCUBE | mesh_type == CYLINDER | mesh_type == SINUSOID, ExcNotImplemented());
        std::srand(314);
        if(mesh_type == HYPERCUBE){
          InitialConditions::Psuedorandom_HYPERCUBE<spacedim> initial_conditions();
        }
        else if(mesh_type == CYLINDER){
          InitialConditions::Psuedorandom_CYLINDER_and_SINUSOID<spacedim> initial_conditions();
        }
        else{
          InitialConditions::Psuedorandom_CYLINDER_and_SINUSOID<spacedim> initial_conditions();
        }
        break;

      case FIXED_RANDOM:
        std::srand(314);
        InitialConditions::Random<spacedim> initial_conditions();
        break;

      case RANDOM:

      
      default:
        break;
      } */

    std::srand(314);
    // InitialConditions::Psuedorandom<spacedim> initial_conditions(mesh_type);

    /* Point<spacedim> hotspot_center;
    if constexpr (MESH==HYPERCUBE){
      if(spacedim==2){
        {
          Point<spacedim> cube_center(0, 0);
          hotspot_center = cube_center;
        }
      }
      else if(spacedim==3){
        {
          Point<spacedim> cube_center(0, 0, 0);
          hotspot_center = cube_center;
        }
      }
      else{
        std::cout << "Trying to use HYPERCUBE with spacedim != 2, 3: This is not implemented" << std::endl;
        throw ExcNotImplemented();
      }
    }

    else if constexpr (MESH == CYLINDER){
      Point<spacedim> cylinder_center(0, 6, 0);
      hotspot_center = cylinder_center;
    }

    else if constexpr (MESH == SPHERE){
      Point<spacedim> sphere_center(18.41988074, 0, 0);
      hotspot_center = sphere_center;
    }

    else if constexpr (MESH == TORUS){
      Point<spacedim> torus_center(13., 0, 0);
      hotspot_center = torus_center;
    }

    else if constexpr (MESH == SINUSOID){
      Point<spacedim> sinusoid_center(0, 9., 0);
      hotspot_center = sinusoid_center;
    } */

    // InitialConditions::HotSpot<spacedim> initial_conditions(hotspot_center);

    /* Function<spacedim> initial_conditions;
    get_initial_values(initial_conditions); */

    InitialCondition<spacedim, MESH, ICTYPE> initial_conditions(r, 0.5);

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

        if(timestep_number%timestep_denominator == 0){
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

        /* {
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
 */
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

        if(timestep_number%timestep_denominator == 0){
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

  template <int dim, int spacedim, MeshType MESH, InitialConditionType ICTYPE>
  void SHEquation<dim, spacedim, MESH, ICTYPE>::make_cylinder()
  {
    Triangulation<3> cylinder;
    GridGenerator::cylinder(cylinder, 6, 18.84955592);

    GridGenerator::extract_boundary_mesh(cylinder, triangulation, {0});

    const CylindricalManifold<dim, spacedim> boundary;
    triangulation.set_all_manifold_ids(0);
    triangulation.set_manifold(0, boundary);

    triangulation.refine_global(refinement_number - 1);
  }

  template <int dim, int spacedim, MeshType MESH, InitialConditionType ICTYPE>
  void SHEquation<dim, spacedim, MESH, ICTYPE>::make_sinusoid()
  {
    Triangulation<3> cylinder;
    GridGenerator::cylinder(cylinder, 6, 18.84955592);

    GridGenerator::extract_boundary_mesh(cylinder, triangulation, {0});

    const CylindricalManifold<dim, spacedim> boundary;
    triangulation.set_all_manifold_ids(0);
    triangulation.set_manifold(0, boundary);

    triangulation.refine_global(refinement_number - 1);

    GridTools::transform(transform_function<spacedim>, triangulation);
  }
  
  template <int dim, int spacedim, MeshType MESH, InitialConditionType ICTYPE>
  void SHEquation<dim, spacedim, MESH, ICTYPE>::make_sphere()
  {
    GridGenerator::hyper_sphere(triangulation, Point<3>(0, 0, 0), 18.41988074);
    triangulation.refine_global(refinement_number - 1);
  }

  template <int dim, int spacedim, MeshType MESH, InitialConditionType ICTYPE>
  void SHEquation<dim, spacedim, MESH, ICTYPE>::make_torus()
  {
    GridGenerator::torus(triangulation, 9., 4.);
    triangulation.refine_global(refinement_number - 2);
  }
  template <int dim, int spacedim, MeshType MESH, InitialConditionType ICTYPE>
  void SHEquation<dim, spacedim, MESH, ICTYPE>::make_hypercube()
  {
    GridGenerator::hyper_cube(triangulation, -18.84955592, 18.84955592);
    triangulation.refine_global(refinement_number);
  }
} // namespace SwiftHohenbergSolver



int main()
{
  /* try
    {
      using namespace SwiftHohenbergSolver;

      SHEquation<2, 3> heat_equation_solver(1, 10, 6, 0.3, 0.0, 1., 80.);
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
    } */

  using namespace SwiftHohenbergSolver;

  MeshType mesh_types[5] = {HYPERCUBE, CYLINDER, SPHERE, TORUS, SINUSOID};
  InitialConditionType ic_types[3] = {HOTSPOT, PSUEDORANDOM, RANDOM};

  const double end_time = 100.;

  const unsigned int ref_num = 6;

  const unsigned int timestep_denominator = 25;

  for(const auto MESH : mesh_types){
    for(const auto ICTYPE: ic_types){
      for(int i = 0; i < 8; ++i){
        const double g_constant = 0.2*i;

        std::cout<< std::endl << std::endl;

        try{
          switch (MESH)
          {
          case HYPERCUBE:
            switch (ICTYPE){
              case HOTSPOT:
              {
                std::string filename = "HYPERCUBE-HOTSPOT-G1-0.2x" + Utilities::int_to_string(i, 1) + "-";
                std::cout << "Running: " << filename << std::endl << std::endl;

                SHEquation<2, 2, HYPERCUBE, HOTSPOT> heat_equation_solver(1, timestep_denominator,
                                                                          ref_num, 0.3, g_constant,
                                                                          filename,  end_time);
                heat_equation_solver.run();
              }
              break;
              
              case PSUEDORANDOM:
              {
                std::string filename = "HYPERCUBE-PSUEDORANDOM-G1-0.2x" + Utilities::int_to_string(i, 1) + "-";
                std::cout << "Running: " << filename << std::endl << std::endl;

                SHEquation<2, 2, HYPERCUBE, PSUEDORANDOM> heat_equation_solver(1, timestep_denominator,
                                                                          ref_num, 0.3, g_constant,
                                                                          filename,  end_time);
                heat_equation_solver.run();
              }
              break;

              case RANDOM:
              {
                std::string filename = "HYPERCUBE-RANDOM-G1-0.2x" + Utilities::int_to_string(i, 1) + "-";
                std::cout << "Running: " << filename << std::endl << std::endl;

                SHEquation<2, 2, HYPERCUBE, RANDOM> heat_equation_solver(1, timestep_denominator,
                                                                          ref_num, 0.3, g_constant,
                                                                          filename,  end_time);
                heat_equation_solver.run();
              }
              break;
            }
            break;
          case CYLINDER:
            switch (ICTYPE){
              case HOTSPOT:
              {
                std::string filename = "CYLINDER-HOTSPOT-G1-0.2x" + Utilities::int_to_string(i, 1) + "-";
                std::cout << "Running: " << filename << std::endl << std::endl;

                SHEquation<2, 3, CYLINDER, HOTSPOT> heat_equation_solver(1, timestep_denominator,
                                                                          ref_num, 0.3, g_constant,
                                                                          filename,  end_time);
                heat_equation_solver.run();
              }
              break;
              
              case PSUEDORANDOM:
              {
                std::string filename = "CYLINDER-PSUEDORANDOM-G1-0.2x" + Utilities::int_to_string(i, 1) + "-";
                std::cout << "Running: " << filename << std::endl << std::endl;

                SHEquation<2, 3, CYLINDER, PSUEDORANDOM> heat_equation_solver(1, timestep_denominator,
                                                                          ref_num, 0.3, g_constant,
                                                                          filename,  end_time);
                heat_equation_solver.run();
              }
              break;

              case RANDOM:
              {
                std::string filename = "CYLINDER-RANDOM-G1-0.2x" + Utilities::int_to_string(i, 1) + "-";
                std::cout << "Running: " << filename << std::endl << std::endl;

                SHEquation<2, 3, CYLINDER, RANDOM> heat_equation_solver(1, timestep_denominator,
                                                                          ref_num, 0.3, g_constant,
                                                                          filename,  end_time);
                heat_equation_solver.run();
              }
              break;
            }
            break;
          case SPHERE:
            switch (ICTYPE){
              case HOTSPOT:
              {
                std::string filename = "SPHERE-HOTSPOT-G1-0.2x" + Utilities::int_to_string(i, 1) + "-";
                std::cout << "Running: " << filename << std::endl << std::endl;

                SHEquation<2, 3, SPHERE, HOTSPOT> heat_equation_solver(1, timestep_denominator,
                                                                          ref_num, 0.3, g_constant,
                                                                          filename,  end_time);
                heat_equation_solver.run();
              }
              break;
              
              case PSUEDORANDOM:
              {
                std::string filename = "SPHERE-PSUEDORANDOM-G1-0.2x" + Utilities::int_to_string(i, 1) + "-";
                std::cout << "Running: " << filename << std::endl << std::endl;

                SHEquation<2, 3, SPHERE, PSUEDORANDOM> heat_equation_solver(1, timestep_denominator,
                                                                          ref_num, 0.3, g_constant,
                                                                          filename,  end_time);
                heat_equation_solver.run();
              }
              break;

              case RANDOM:
              {
                std::string filename = "SPHERE-RANDOM-G1-0.2x" + Utilities::int_to_string(i, 1) + "-";
                std::cout << "Running: " << filename << std::endl << std::endl;

                SHEquation<2, 3, SPHERE, RANDOM> heat_equation_solver(1, timestep_denominator,
                                                                          ref_num, 0.3, g_constant,
                                                                          filename,  end_time);
                heat_equation_solver.run();
              }
              break;
            }
            break;
          case TORUS:
            switch (ICTYPE){
              case HOTSPOT:
              {
                std::string filename = "TORUS-HOTSPOT-G1-0.2x" + Utilities::int_to_string(i, 1) + "-";
                std::cout << "Running: " << filename << std::endl << std::endl;

                SHEquation<2, 3, TORUS, HOTSPOT> heat_equation_solver(1, timestep_denominator,
                                                                          ref_num, 0.3, g_constant,
                                                                          filename,  end_time);
                heat_equation_solver.run();
              }
              break;
              
              case PSUEDORANDOM:
              {
                std::string filename = "TORUS-PSUEDORANDOM-G1-0.2x" + Utilities::int_to_string(i, 1) + "-";
                std::cout << "Running: " << filename << std::endl << std::endl;

                SHEquation<2, 3, TORUS, PSUEDORANDOM> heat_equation_solver(1, timestep_denominator,
                                                                          ref_num, 0.3, g_constant,
                                                                          filename,  end_time);
                heat_equation_solver.run();
              }
              break;

              case RANDOM:
              {
                std::string filename = "TORUS-RANDOM-G1-0.2x" + Utilities::int_to_string(i, 1) + "-";
                std::cout << "Running: " << filename << std::endl << std::endl;

                SHEquation<2, 3, TORUS, RANDOM> heat_equation_solver(1, timestep_denominator,
                                                                          ref_num, 0.3, g_constant,
                                                                          filename,  end_time);
                heat_equation_solver.run();
              }
              break;
            }
            break;
          case SINUSOID:
            switch (ICTYPE){
              case HOTSPOT:
              {
                std::string filename = "SINUSOID-HOTSPOT-G1-0.2x" + Utilities::int_to_string(i, 1) + "-";
                std::cout << "Running: " << filename << std::endl << std::endl;

                SHEquation<2, 3, SINUSOID, HOTSPOT> heat_equation_solver(1, timestep_denominator,
                                                                          ref_num, 0.3, g_constant,
                                                                          filename,  end_time);
                heat_equation_solver.run();
              }
              break;
              
              case PSUEDORANDOM:
              {
                std::string filename = "SINUSOID-PSUEDORANDOM-G1-0.2x" + Utilities::int_to_string(i, 1) + "-";
                std::cout << "Running: " << filename << std::endl << std::endl;

                SHEquation<2, 3, SINUSOID, PSUEDORANDOM> heat_equation_solver(1, timestep_denominator,
                                                                          ref_num, 0.3, g_constant,
                                                                          filename,  end_time);
                heat_equation_solver.run();
              }
              break;

              case RANDOM:
              {
                std::string filename = "SINUSOID-RANDOM-G1-0.2x" + Utilities::int_to_string(i, 1) + "-";
                std::cout << "Running: " << filename << std::endl << std::endl;

                SHEquation<2, 3, SINUSOID, RANDOM> heat_equation_solver(1, timestep_denominator,
                                                                          ref_num, 0.3, g_constant,
                                                                          filename,  end_time);
                heat_equation_solver.run();
              }
              break;
            }
            break;
          default:
            break;
          }
        }
        catch (std::exception &exc)
        {
          std::cout << "An error occured" << std::endl;
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
          std::cout << "Error occured, made it past first catch" << std::endl;
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

  /* using namespace SwiftHohenbergSolver;
  MeshType mesh_types[5] = {HYPERCUBE, CYLINDER, SPHERE, TORUS, SINUSOID};

  double end_time = 1;

  unsigned int ref_num = 6;


  for(int i = 0; i < 5; ++i){
    try
    {
      switch (mesh_types[i])
      {
      case HYPERCUBE:
        {
        const double g_constant = 0.2*i;

        const unsigned int timestep_denominator = 25;

        std::cout  << std::endl << "Running simulation with mesh HYPERCUBE" << std::endl;

        std::string filename = "HOTSPOT-TEST-HYPERCUBE-";

        SHEquation<2, 2, HYPERCUBE> heat_equation_solver(1, timestep_denominator, ref_num, 0.3, g_constant, filename, HOTSPOT, end_time);
        heat_equation_solver.run();
        }
        break;

      case CYLINDER:
        {const double g_constant = 0.2*i;

        const unsigned int timestep_denominator = 25;

        std::cout  << std::endl << "Running simulation with mesh CYLINDER" << std::endl;

        std::string filename = "HOTSPOT-TEST-CYLINDER-";

        SHEquation<2, 3, CYLINDER> heat_equation_solver(1, timestep_denominator, ref_num, 0.3, g_constant, filename, HOTSPOT, end_time);
        heat_equation_solver.run();
        }
        break;
      
      case SINUSOID:
        {
        const double g_constant = 0;

        const unsigned int timestep_denominator = 25;

        std::cout  << std::endl << "Running simulation with mesh SINUSOID" << std::endl;

        std::string filename = "HOTSPOT-TEST-SINUSOID-";

        SHEquation<2, 3, SINUSOID> heat_equation_solver(1, timestep_denominator, ref_num, 0.3, g_constant, filename, HOTSPOT, end_time);
        heat_equation_solver.run();
        }
        break;
      
      case SPHERE:
      {
        const double g_constant = 0;

        const unsigned int timestep_denominator = 25;

        std::cout  << std::endl << "Running simulation with mesh SPHERE" << std::endl;

        std::string filename = "HOTSPOT-TEST-SPHERE-";

        SHEquation<2, 3, SPHERE> heat_equation_solver(1, timestep_denominator, ref_num, 0.3, g_constant, filename, HOTSPOT, end_time);
        heat_equation_solver.run();
        break;
      }

      case TORUS:
      {
        const double g_constant = 0;

        const unsigned int timestep_denominator = 25;

        std::cout  << std::endl << "Running simulation with mesh TORUS" << std::endl;

        std::string filename = "HOTSPOT-TEST-TORUS-";

        SHEquation<2, 3, TORUS> heat_equation_solver(1, timestep_denominator, ref_num, 0.3, g_constant, filename, HOTSPOT, end_time);
        heat_equation_solver.run();
        break;
      }
      
      default:
        std::cout << "An error occured" << std::endl;
        throw ExcNotImplemented();
        break;
      }
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

  } */

  /* for(unsigned int i = 3; i < 7; ++i){
    for(unsigned int j = 1; j<21; ++j){
      std::cout << std::endl;
      std::cout << "=============================================================" << std::endl;
      std::cout << "Running step-26 with refinement " << i << " and timestep 1 over " << j*200 << std::endl << std::endl;
      {
        try
          {
            using namespace SwiftHohenbergSolver;

            SHEquation<2, 2> heat_equation_solver(j*200, i);
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
