cmake_minimum_required( VERSION 3.26.3 )

# Set version number
set( SEM_KERNELS_VERSION_MAJOR 0 )
set( SEM_KERNELS_VERSION_MINOR 1 )
set( SEM_KERNELS_VERSION_PATCHLEVEL 0 )

# check if SEMKernels is build as a submodule or a separate project
get_directory_property( parent_dir PARENT_DIRECTORY )

add_subdirectory( src )

if( SEMKERNELS_ENABLE_DOCS )
#    add_subdirectory( docs )
endif()


