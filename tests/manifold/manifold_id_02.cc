// ---------------------------------------------------------------------
//
// Copyright (C) 2011 - 2020 by the deal.II authors
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


// Test Manifold ID. Now we test the function set_all_manifold_ids(), and verify
// that they are correctly inherited from one cell onward. All manifold ids
// should end up being 1.

#include "../tests.h"


// all include files you need here
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

// Helper function
template <int dim, int spacedim>
void
test(unsigned int ref = 1)
{
  deallog << "Testing dim=" << dim << ", spacedim=" << spacedim << std::endl;

  Triangulation<dim, spacedim> tria;
  GridGenerator::hyper_cube(tria);
  Point<spacedim> center;
  for (unsigned int i = 0; i < dim; ++i)
    center[i] = .25;

  // const SphericalManifold<dim,spacedim> boundary(center,center.norm());
  // triangulation.set_manifold (0, boundary_description);

  typename Triangulation<dim, spacedim>::active_cell_iterator cell;

  tria.begin_active()->set_all_manifold_ids(1);
  tria.refine_global(ref);

  for (cell = tria.begin_active(); cell != tria.end(); ++cell)
    {
      deallog << "C: " << cell << ", mid: " << (int)cell->manifold_id()
              << std::endl;
      for (const unsigned int f : GeometryInfo<dim>::face_indices())
        deallog << "f: " << cell->face(f)
                << ", mid: " << (int)cell->face(f)->manifold_id() << std::endl;
    }
}

int
main()
{
  initlog();

  test<1, 1>();
  test<1, 2>();
  test<2, 2>();
  test<2, 3>();
  test<3, 3>();

  return 0;
}