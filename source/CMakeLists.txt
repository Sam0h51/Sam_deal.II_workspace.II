## ---------------------------------------------------------------------
##
## Copyright (C) 2012 - 2022 by the deal.II authors
##
## This file is part of the deal.II library.
##
## The deal.II library is free software; you can use it, redistribute
## it, and/or modify it under the terms of the GNU Lesser General
## Public License as published by the Free Software Foundation; either
## version 2.1 of the License, or (at your option) any later version.
## The full text of the license can be found in the file LICENSE.md at
## the top level directory of deal.II.
##
## ---------------------------------------------------------------------

MESSAGE(STATUS "Setting up library")

#
# This happens surprisingly often:
#
IF("${DEAL_II_PACKAGE_NAME}" STREQUAL  "")
  MESSAGE(FATAL_ERROR
    "\nWrong source directory specified.\n"
    "You pointed cmake to the subdirectory \"source\" instead of the base "
    "directory (the one containing the file \"README.md\"). Make sure that "
    "the path in the cmake invocation does not end in \".../source\". If you "
    "have created a \"build\" subdirectory, the cmake invocation should simply "
    "be of the form:\n"
    "  $ cmake  [...options...]  ../\n"
    "(path not ending in \".../source\")"
    )
ENDIF()

#
# Compile the deal.II library
#

INCLUDE_DIRECTORIES(
  ${CMAKE_BINARY_DIR}/include/
  ${CMAKE_SOURCE_DIR}/include/
  ${DEAL_II_BUNDLED_INCLUDE_DIRS}
  SYSTEM
  ${DEAL_II_INCLUDE_DIRS}
  )

#
# List the directories where we have source files. the ones with the longest
# compile jobs come first so that 'make -j N' saturates many processors also
# towards the end of compiling rather than having to wait for one long
# compilation that, because it has been listed last, is started towards the
# end of everything (e.g. numerics/vectors.cc takes several minutes to
# compile...)
#
ADD_SUBDIRECTORY(numerics)
ADD_SUBDIRECTORY(fe)
ADD_SUBDIRECTORY(dofs)
ADD_SUBDIRECTORY(lac)
ADD_SUBDIRECTORY(base)
ADD_SUBDIRECTORY(cgal)
ADD_SUBDIRECTORY(gmsh)
ADD_SUBDIRECTORY(grid)
ADD_SUBDIRECTORY(hp)
ADD_SUBDIRECTORY(multigrid)
ADD_SUBDIRECTORY(distributed)
ADD_SUBDIRECTORY(algorithms)
ADD_SUBDIRECTORY(integrators)
ADD_SUBDIRECTORY(matrix_free)
ADD_SUBDIRECTORY(meshworker)
ADD_SUBDIRECTORY(opencascade)
ADD_SUBDIRECTORY(particles)
ADD_SUBDIRECTORY(differentiation)
ADD_SUBDIRECTORY(physics)
ADD_SUBDIRECTORY(optimization/rol)
ADD_SUBDIRECTORY(non_matching)
ADD_SUBDIRECTORY(sundials)
ADD_SUBDIRECTORY(arborx)

FOREACH(build ${DEAL_II_BUILD_TYPES})
  STRING(TOLOWER ${build} build_lowercase)

  #
  # Combine all ${build} OBJECT targets to a ${build} library:
  #

  GET_PROPERTY(_objects GLOBAL PROPERTY DEAL_II_OBJECTS_${build})

  #
  # FIXME: Somehow the cuda object files lose the "generated" property.
  # This is an ugly hack
  #
  FOREACH(_object ${_objects})
    IF("${_object}" MATCHES "cu.o$")
      SET_SOURCE_FILES_PROPERTIES("${_object}"
        PROPERTIES
        EXTERNAL_OBJECT TRUE
        GENERATED TRUE
        )
    ENDIF()
  ENDFOREACH()

  ADD_LIBRARY(${DEAL_II_BASE_NAME}${DEAL_II_${build}_SUFFIX}
    dummy.cc # Workaround for a bug in the Xcode generator
    ${_objects}
    )
  ADD_DEPENDENCIES(library ${DEAL_II_BASE_NAME}${DEAL_II_${build}_SUFFIX})

  SET_TARGET_PROPERTIES(${DEAL_II_BASE_NAME}${DEAL_II_${build}_SUFFIX}
    PROPERTIES
    VERSION "${DEAL_II_PACKAGE_VERSION}"
    #
    # Sonaming: Well... we just use the version number.
    # No point to wrack one's brain over the question whether a new version of
    # a C++ library is still ABI backwards compatible :-]
    #
    SOVERSION "${DEAL_II_PACKAGE_VERSION}"
    LINK_FLAGS "${DEAL_II_LINKER_FLAGS} ${DEAL_II_LINKER_FLAGS_${build}}"
    LINKER_LANGUAGE "CXX"
    COMPILE_DEFINITIONS "${DEAL_II_DEFINITIONS};${DEAL_II_DEFINITIONS_${build}}"
    COMPILE_FLAGS "${DEAL_II_CXX_FLAGS} ${DEAL_II_CXX_FLAGS_${build}}"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${DEAL_II_LIBRARY_RELDIR}"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${DEAL_II_LIBRARY_RELDIR}"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${DEAL_II_EXECUTABLE_RELDIR}"
    )

  IF(DEAL_II_WITH_CUDA)
    # We need to use the CUDA linker to avoid confusing nvlink with sonaming.
    SET_TARGET_PROPERTIES(${DEAL_II_BASE_NAME}${DEAL_II_${build}_SUFFIX}
    PROPERTIES
    LINKER_LANGUAGE "CUDA"
    )
  ENDIF()

  IF(CMAKE_SYSTEM_NAME MATCHES "Darwin")
    SET_TARGET_PROPERTIES(${DEAL_II_BASE_NAME}${DEAL_II_${build}_SUFFIX}
      PROPERTIES
      MACOSX_RPATH OFF
      BUILD_WITH_INSTALL_RPATH OFF
      INSTALL_NAME_DIR "${CMAKE_INSTALL_PREFIX}/${DEAL_II_LIBRARY_RELDIR}"
      )
  ENDIF()

  # Under Windows (MSVC) cmake will always generate multi-configuration
  # projects. When building on the command line with 'cmake --build .',
  # release and debug builds of the library are done with the default 'Debug'
  # configuration. This causes the debug and release .lib to be built inside
  # ./lib/Debug/. This is not very pretty and confuses example/test projects,
  # so we just hard-wire the location here. We only really need to set static
  # lib locations for _DEBUG (no support for dynamic linking, _RELEASE will be
  # ignored), but we do it anyhow.
  IF (DEAL_II_MSVC)
    SET_TARGET_PROPERTIES(${DEAL_II_BASE_NAME}${DEAL_II_${build}_SUFFIX}
      PROPERTIES
      ARCHIVE_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/${DEAL_II_LIBRARY_RELDIR}"
      LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/${DEAL_II_LIBRARY_RELDIR}"
      RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/${DEAL_II_EXECUTABLE_RELDIR}"
      ARCHIVE_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/${DEAL_II_LIBRARY_RELDIR}"
      LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/${DEAL_II_LIBRARY_RELDIR}"
      RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/${DEAL_II_EXECUTABLE_RELDIR}"
      )
  ENDIF()


  TARGET_LINK_LIBRARIES(${DEAL_II_BASE_NAME}${DEAL_II_${build}_SUFFIX}
    ${DEAL_II_LIBRARIES_${build}}
    ${DEAL_II_LIBRARIES}
    )

  FILE(MAKE_DIRECTORY
    ${CMAKE_BINARY_DIR}/${DEAL_II_PROJECT_CONFIG_RELDIR}
    )
  EXPORT(TARGETS ${DEAL_II_BASE_NAME}${DEAL_II_${build}_SUFFIX}
    FILE ${CMAKE_BINARY_DIR}/${DEAL_II_PROJECT_CONFIG_RELDIR}/${DEAL_II_PROJECT_CONFIG_NAME}Targets.cmake
    APPEND
    )

  INSTALL(TARGETS ${DEAL_II_BASE_NAME}${DEAL_II_${build}_SUFFIX}
    COMPONENT library
    EXPORT ${DEAL_II_PROJECT_CONFIG_NAME}Targets
    RUNTIME DESTINATION ${DEAL_II_EXECUTABLE_RELDIR}
    LIBRARY DESTINATION ${DEAL_II_LIBRARY_RELDIR}
    ARCHIVE DESTINATION ${DEAL_II_LIBRARY_RELDIR}
    )
ENDFOREACH()

INSTALL(EXPORT ${DEAL_II_PROJECT_CONFIG_NAME}Targets
  DESTINATION ${DEAL_II_PROJECT_CONFIG_RELDIR}
  COMPONENT library
  )

MESSAGE(STATUS "Setting up library - Done")
