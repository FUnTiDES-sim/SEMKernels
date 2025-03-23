cmake_minimum_required( VERSION 3.26.3 )

# Set version number
set( SEM_KERNELS_VERSION_MAJOR 0 )
set( SEM_KERNELS_VERSION_MINOR 1 )
set( SEM_KERNELS_VERSION_PATCHLEVEL 0 )

# check if SEMKernels is build as a submodule or a separate project
get_directory_property( parent_dir PARENT_DIRECTORY )
if(parent_dir)
    set( is_submodule ON )
else()
    set( is_submodule OFF )
endif()

if( is_submodule )
if( NOT BLT_LOADED )
    message( FATAL_ERROR "When using SEMKernels as a submodule you must have already loaded BLT." )
    endif()

    option( SEMKERNELS_ENABLE_UNIT_TESTS "Builds tests" OFF )
    option( SEMKERNELS_ENABLE_EXAMPLES "Builds examples" OFF )
    option( SEMKERNELS_ENABLE_BENCHMARKS "Builds benchmarks" OFF )
    option( SEMKERNELS_ENABLE_DOCS "Builds documentation" OFF )

    include( cmake/CMakeBasics.cmake )
else()
    message( "not a submodule")
    project( SEMKernels LANGUAGES CXX C )

    set( BLT_CXX_STD "c++17" CACHE STRING "Version of C++ standard" FORCE )
    set( ENABLE_WARNINGS_AS_ERRORS "ON" CACHE PATH "" )

    option( SEMKERNELS_ENABLE_UNIT_TESTS "Builds tests" ON )
    option( SEMKERNELS_ENABLE_EXAMPLES "Builds examples" ON )
    option( SEMKERNELS_ENABLE_BENCHMARKS "Builds benchmarks" ON )
    option( SEMKERNELS_ENABLE_DOCS "Builds documentation" ON )

    option( ENABLE_CUDA "Build with CUDA" OFF )
    option( ENABLE_HIP "Build with HIP" OFF )

    if( NOT BLT_LOADED )
        if( DEFINED BLT_SOURCE_DIR )
            if( NOT EXISTS ${BLT_SOURCE_DIR}/SetupBLT.cmake )
                message( FATAL_ERROR "Given BLT_SOURCE_DIR does not contain SetupBLT.cmake" )
            endif()
        else ()
            set( BLT_SOURCE_DIR ${PROJECT_SOURCE_DIR}/cmake/blt CACHE PATH "" )

            if( NOT EXISTS ${BLT_SOURCE_DIR}/SetupBLT.cmake )
                message( FATAL_ERROR "The BLT submodule is not present. If in git repository run the following two commands:\n \
                git submodule init\n \
                git submodule update" )
            endif ()
        endif ()

        include( ${BLT_SOURCE_DIR}/SetupBLT.cmake )
    endif()

    include( cmake/CMakeBasics.cmake )
endif()



add_subdirectory( src )

if( SEMKERNELS_ENABLE_DOCS )
#    add_subdirectory( docs )
endif()


