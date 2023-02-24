// ---------------------------------------------------------------------
//
// Copyright (C) 2021 - 2021 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE.md at
// the top level directory of deal.II.
//
// ---------------------------------------------------------------------

/*
 * Test that QuadratureGenerator clears its previously created quadratures
 * when we call generate() again.
 */

#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/hp/q_collection.h>

#include "deal.II/non_matching/quadrature_generator.h"

#include "../tests.h"


// Print the sizes of all the quadratures that QuadratureGenerator creates to
// deallog.
template <int dim>
void
print_n_quadrature_points(
  const NonMatching::QuadratureGenerator<dim> &quadrature_generator)
{
  deallog << "inside " << quadrature_generator.get_inside_quadrature().size()
          << std::endl;

  deallog << "outside " << quadrature_generator.get_outside_quadrature().size()
          << std::endl;

  deallog << "surface " << quadrature_generator.get_surface_quadrature().size()
          << std::endl;

  deallog << std::endl;
}



// Call the QuadratureGenerator::generate with the same level set function
// twice. Make sure that the sizes of the constructed quadratures are the same
// both times. The purpose is to make sure that the previously created
// quadratures have been cleared before we create the new ones.
template <int dim>
void
test()
{
  deallog << "dim = " << dim << std::endl;

  const hp::QCollection<1>              q_collection(QGauss<1>(1));
  NonMatching::QuadratureGenerator<dim> quadrature_generator(q_collection);

  const Functions::ConstantFunction<dim> level_set(1);

  const BoundingBox<dim> box = create_unit_bounding_box<dim>();

  deallog << "quadrature sizes first call" << std::endl;
  quadrature_generator.generate(level_set, box);
  print_n_quadrature_points(quadrature_generator);

  deallog << "quadrature sizes second call" << std::endl;
  quadrature_generator.generate(level_set, box);
  print_n_quadrature_points(quadrature_generator);
}



int
main()
{
  initlog();

  test<1>();
  test<2>();
  test<3>();
}
