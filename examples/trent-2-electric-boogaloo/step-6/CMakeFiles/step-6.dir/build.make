# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/ubuntu/deal.II/dealii/examples/trent-2-electric-boogaloo/step-6

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ubuntu/deal.II/dealii/examples/trent-2-electric-boogaloo/step-6

# Include any dependencies generated for this target.
include CMakeFiles/step-6.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/step-6.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/step-6.dir/flags.make

CMakeFiles/step-6.dir/step-6.cc.o: CMakeFiles/step-6.dir/flags.make
CMakeFiles/step-6.dir/step-6.cc.o: step-6.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ubuntu/deal.II/dealii/examples/trent-2-electric-boogaloo/step-6/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/step-6.dir/step-6.cc.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/step-6.dir/step-6.cc.o -c /home/ubuntu/deal.II/dealii/examples/trent-2-electric-boogaloo/step-6/step-6.cc

CMakeFiles/step-6.dir/step-6.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/step-6.dir/step-6.cc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ubuntu/deal.II/dealii/examples/trent-2-electric-boogaloo/step-6/step-6.cc > CMakeFiles/step-6.dir/step-6.cc.i

CMakeFiles/step-6.dir/step-6.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/step-6.dir/step-6.cc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ubuntu/deal.II/dealii/examples/trent-2-electric-boogaloo/step-6/step-6.cc -o CMakeFiles/step-6.dir/step-6.cc.s

# Object files for target step-6
step__6_OBJECTS = \
"CMakeFiles/step-6.dir/step-6.cc.o"

# External object files for target step-6
step__6_EXTERNAL_OBJECTS =

step-6: CMakeFiles/step-6.dir/step-6.cc.o
step-6: CMakeFiles/step-6.dir/build.make
step-6: /home/ubuntu/deal.II/installed/lib/libdeal_II.so.9.4.0
step-6: /home/ubuntu/libs/p4est-2.3.2/FAST/lib/libp4est.so
step-6: /home/ubuntu/libs/p4est-2.3.2/FAST/lib/libsc.so
step-6: /usr/lib/x86_64-linux-gnu/openmpi/lib/libopen-pal.so
step-6: /home/ubuntu/libs/trilinos-release-12-18-1/lib/librol.so
step-6: /home/ubuntu/libs/trilinos-release-12-18-1/lib/libtempus.so
step-6: /home/ubuntu/libs/trilinos-release-12-18-1/lib/libmuelu-adapters.so
step-6: /home/ubuntu/libs/trilinos-release-12-18-1/lib/libmuelu-interface.so
step-6: /home/ubuntu/libs/trilinos-release-12-18-1/lib/libmuelu.so
step-6: /home/ubuntu/libs/trilinos-release-12-18-1/lib/liblocathyra.so
step-6: /home/ubuntu/libs/trilinos-release-12-18-1/lib/liblocaepetra.so
step-6: /home/ubuntu/libs/trilinos-release-12-18-1/lib/liblocalapack.so
step-6: /home/ubuntu/libs/trilinos-release-12-18-1/lib/libloca.so
step-6: /home/ubuntu/libs/trilinos-release-12-18-1/lib/libnoxepetra.so
step-6: /home/ubuntu/libs/trilinos-release-12-18-1/lib/libnoxlapack.so
step-6: /home/ubuntu/libs/trilinos-release-12-18-1/lib/libnox.so
step-6: /home/ubuntu/libs/trilinos-release-12-18-1/lib/libintrepid2.so
step-6: /home/ubuntu/libs/trilinos-release-12-18-1/lib/libintrepid.so
step-6: /home/ubuntu/libs/trilinos-release-12-18-1/lib/libteko.so
step-6: /home/ubuntu/libs/trilinos-release-12-18-1/lib/libstratimikos.so
step-6: /home/ubuntu/libs/trilinos-release-12-18-1/lib/libstratimikosbelos.so
step-6: /home/ubuntu/libs/trilinos-release-12-18-1/lib/libstratimikosamesos2.so
step-6: /home/ubuntu/libs/trilinos-release-12-18-1/lib/libstratimikosaztecoo.so
step-6: /home/ubuntu/libs/trilinos-release-12-18-1/lib/libstratimikosamesos.so
step-6: /home/ubuntu/libs/trilinos-release-12-18-1/lib/libstratimikosml.so
step-6: /home/ubuntu/libs/trilinos-release-12-18-1/lib/libstratimikosifpack.so
step-6: /home/ubuntu/libs/trilinos-release-12-18-1/lib/libanasazitpetra.so
step-6: /home/ubuntu/libs/trilinos-release-12-18-1/lib/libModeLaplace.so
step-6: /home/ubuntu/libs/trilinos-release-12-18-1/lib/libanasaziepetra.so
step-6: /home/ubuntu/libs/trilinos-release-12-18-1/lib/libanasazi.so
step-6: /home/ubuntu/libs/trilinos-release-12-18-1/lib/libamesos2.so
step-6: /home/ubuntu/libs/trilinos-release-12-18-1/lib/libshylu_nodetacho.so
step-6: /home/ubuntu/libs/trilinos-release-12-18-1/lib/libbelosxpetra.so
step-6: /home/ubuntu/libs/trilinos-release-12-18-1/lib/libbelostpetra.so
step-6: /home/ubuntu/libs/trilinos-release-12-18-1/lib/libbelosepetra.so
step-6: /home/ubuntu/libs/trilinos-release-12-18-1/lib/libbelos.so
step-6: /home/ubuntu/libs/trilinos-release-12-18-1/lib/libml.so
step-6: /home/ubuntu/libs/trilinos-release-12-18-1/lib/libifpack.so
step-6: /home/ubuntu/libs/trilinos-release-12-18-1/lib/libzoltan2.so
step-6: /home/ubuntu/libs/trilinos-release-12-18-1/lib/libpamgen_extras.so
step-6: /home/ubuntu/libs/trilinos-release-12-18-1/lib/libpamgen.so
step-6: /home/ubuntu/libs/trilinos-release-12-18-1/lib/libamesos.so
step-6: /home/ubuntu/libs/trilinos-release-12-18-1/lib/libgaleri-xpetra.so
step-6: /home/ubuntu/libs/trilinos-release-12-18-1/lib/libgaleri-epetra.so
step-6: /home/ubuntu/libs/trilinos-release-12-18-1/lib/libaztecoo.so
step-6: /home/ubuntu/libs/trilinos-release-12-18-1/lib/libisorropia.so
step-6: /home/ubuntu/libs/trilinos-release-12-18-1/lib/libxpetra-sup.so
step-6: /home/ubuntu/libs/trilinos-release-12-18-1/lib/libxpetra.so
step-6: /home/ubuntu/libs/trilinos-release-12-18-1/lib/libthyratpetra.so
step-6: /home/ubuntu/libs/trilinos-release-12-18-1/lib/libthyraepetraext.so
step-6: /home/ubuntu/libs/trilinos-release-12-18-1/lib/libthyraepetra.so
step-6: /home/ubuntu/libs/trilinos-release-12-18-1/lib/libthyracore.so
step-6: /home/ubuntu/libs/trilinos-release-12-18-1/lib/libtrilinosss.so
step-6: /home/ubuntu/libs/trilinos-release-12-18-1/lib/libtpetraext.so
step-6: /home/ubuntu/libs/trilinos-release-12-18-1/lib/libtpetrainout.so
step-6: /home/ubuntu/libs/trilinos-release-12-18-1/lib/libtpetra.so
step-6: /home/ubuntu/libs/trilinos-release-12-18-1/lib/libkokkostsqr.so
step-6: /home/ubuntu/libs/trilinos-release-12-18-1/lib/libtpetraclassiclinalg.so
step-6: /home/ubuntu/libs/trilinos-release-12-18-1/lib/libtpetraclassicnodeapi.so
step-6: /home/ubuntu/libs/trilinos-release-12-18-1/lib/libtpetraclassic.so
step-6: /home/ubuntu/libs/trilinos-release-12-18-1/lib/libepetraext.so
step-6: /home/ubuntu/libs/trilinos-release-12-18-1/lib/libtriutils.so
step-6: /home/ubuntu/libs/trilinos-release-12-18-1/lib/libshards.so
step-6: /home/ubuntu/libs/trilinos-release-12-18-1/lib/libzoltan.so
step-6: /home/ubuntu/libs/trilinos-release-12-18-1/lib/libepetra.so
step-6: /home/ubuntu/libs/trilinos-release-12-18-1/lib/libsacado.so
step-6: /home/ubuntu/libs/trilinos-release-12-18-1/lib/librtop.so
step-6: /home/ubuntu/libs/trilinos-release-12-18-1/lib/libkokkoskernels.so
step-6: /home/ubuntu/libs/trilinos-release-12-18-1/lib/libteuchoskokkoscomm.so
step-6: /home/ubuntu/libs/trilinos-release-12-18-1/lib/libteuchoskokkoscompat.so
step-6: /home/ubuntu/libs/trilinos-release-12-18-1/lib/libteuchosremainder.so
step-6: /home/ubuntu/libs/trilinos-release-12-18-1/lib/libteuchosnumerics.so
step-6: /home/ubuntu/libs/trilinos-release-12-18-1/lib/libteuchoscomm.so
step-6: /home/ubuntu/libs/trilinos-release-12-18-1/lib/libteuchosparameterlist.so
step-6: /home/ubuntu/libs/trilinos-release-12-18-1/lib/libteuchosparser.so
step-6: /home/ubuntu/libs/trilinos-release-12-18-1/lib/libteuchoscore.so
step-6: /home/ubuntu/libs/trilinos-release-12-18-1/lib/libkokkosalgorithms.so
step-6: /home/ubuntu/libs/trilinos-release-12-18-1/lib/libkokkoscontainers.so
step-6: /home/ubuntu/libs/trilinos-release-12-18-1/lib/libkokkoscore.so
step-6: /home/ubuntu/libs/trilinos-release-12-18-1/lib/libgtest.so
step-6: /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi_cxx.so
step-6: /usr/lib/x86_64-linux-gnu/libarpack.so
step-6: /home/ubuntu/libs/hdf5-1.10.7/lib/libhdf5.so
step-6: /usr/lib/x86_64-linux-gnu/libz.so
step-6: /home/ubuntu/libs/oce-OCE-0.18.3/lib/libTKBO.so
step-6: /home/ubuntu/libs/oce-OCE-0.18.3/lib/libTKBool.so
step-6: /home/ubuntu/libs/oce-OCE-0.18.3/lib/libTKBRep.so
step-6: /home/ubuntu/libs/oce-OCE-0.18.3/lib/libTKernel.so
step-6: /home/ubuntu/libs/oce-OCE-0.18.3/lib/libTKFeat.so
step-6: /home/ubuntu/libs/oce-OCE-0.18.3/lib/libTKFillet.so
step-6: /home/ubuntu/libs/oce-OCE-0.18.3/lib/libTKG2d.so
step-6: /home/ubuntu/libs/oce-OCE-0.18.3/lib/libTKG3d.so
step-6: /home/ubuntu/libs/oce-OCE-0.18.3/lib/libTKGeomAlgo.so
step-6: /home/ubuntu/libs/oce-OCE-0.18.3/lib/libTKGeomBase.so
step-6: /home/ubuntu/libs/oce-OCE-0.18.3/lib/libTKHLR.so
step-6: /home/ubuntu/libs/oce-OCE-0.18.3/lib/libTKIGES.so
step-6: /home/ubuntu/libs/oce-OCE-0.18.3/lib/libTKMath.so
step-6: /home/ubuntu/libs/oce-OCE-0.18.3/lib/libTKMesh.so
step-6: /home/ubuntu/libs/oce-OCE-0.18.3/lib/libTKOffset.so
step-6: /home/ubuntu/libs/oce-OCE-0.18.3/lib/libTKPrim.so
step-6: /home/ubuntu/libs/oce-OCE-0.18.3/lib/libTKShHealing.so
step-6: /home/ubuntu/libs/oce-OCE-0.18.3/lib/libTKSTEP.so
step-6: /home/ubuntu/libs/oce-OCE-0.18.3/lib/libTKSTEPAttr.so
step-6: /home/ubuntu/libs/oce-OCE-0.18.3/lib/libTKSTEPBase.so
step-6: /home/ubuntu/libs/oce-OCE-0.18.3/lib/libTKSTEP209.so
step-6: /home/ubuntu/libs/oce-OCE-0.18.3/lib/libTKSTL.so
step-6: /home/ubuntu/libs/oce-OCE-0.18.3/lib/libTKTopAlgo.so
step-6: /home/ubuntu/libs/oce-OCE-0.18.3/lib/libTKXSBase.so
step-6: /home/ubuntu/libs/slepc-3.13.2/lib/libslepc.so
step-6: /home/ubuntu/libs/petsc-3.13.1/lib/libpetsc.so
step-6: /home/ubuntu/libs/petsc-3.13.1/lib/libHYPRE.so
step-6: /home/ubuntu/libs/petsc-3.13.1/lib/libcmumps.a
step-6: /home/ubuntu/libs/petsc-3.13.1/lib/libdmumps.a
step-6: /home/ubuntu/libs/petsc-3.13.1/lib/libsmumps.a
step-6: /home/ubuntu/libs/petsc-3.13.1/lib/libzmumps.a
step-6: /home/ubuntu/libs/petsc-3.13.1/lib/libmumps_common.a
step-6: /home/ubuntu/libs/petsc-3.13.1/lib/libpord.a
step-6: /home/ubuntu/libs/petsc-3.13.1/lib/libscalapack.a
step-6: /usr/lib/x86_64-linux-gnu/liblapack.so
step-6: /usr/lib/x86_64-linux-gnu/libblas.so
step-6: /home/ubuntu/libs/parmetis-4.0.3/lib/libparmetis.so
step-6: /home/ubuntu/libs/parmetis-4.0.3/lib/libmetis.so
step-6: /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi_usempif08.so
step-6: /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi_usempi_ignore_tkr.so
step-6: /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi_mpifh.so
step-6: /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi.so
step-6: CMakeFiles/step-6.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/ubuntu/deal.II/dealii/examples/trent-2-electric-boogaloo/step-6/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable step-6"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/step-6.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/step-6.dir/build: step-6

.PHONY : CMakeFiles/step-6.dir/build

CMakeFiles/step-6.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/step-6.dir/cmake_clean.cmake
.PHONY : CMakeFiles/step-6.dir/clean

CMakeFiles/step-6.dir/depend:
	cd /home/ubuntu/deal.II/dealii/examples/trent-2-electric-boogaloo/step-6 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ubuntu/deal.II/dealii/examples/trent-2-electric-boogaloo/step-6 /home/ubuntu/deal.II/dealii/examples/trent-2-electric-boogaloo/step-6 /home/ubuntu/deal.II/dealii/examples/trent-2-electric-boogaloo/step-6 /home/ubuntu/deal.II/dealii/examples/trent-2-electric-boogaloo/step-6 /home/ubuntu/deal.II/dealii/examples/trent-2-electric-boogaloo/step-6/CMakeFiles/step-6.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/step-6.dir/depend

