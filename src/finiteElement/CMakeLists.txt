
set( finiteElement_headers
     SEMQkGLBasisFunctions.hpp
     SEMQkGLIntegrals.hpp
     classic/SEMQkGLIntegrals.hpp
     optim/SEMQkGLIntegralsOptim.hpp
     shiva/SEMQkGLIntegralsShiva.hpp
     )

set( finiteElement_sources
     )

set( finiteElement_dependencies
     shiva
     )

if( ENABLE_HIP )
    list( APPEND finiteElement_dependencies blt::hip camp )
endif()

if( ENABLE_CUDA )
    list( APPEND finiteElement_dependencies cuda )
endif()

blt_add_library( NAME             finiteElement
#                 SOURCES          ${finiteElement_sources}
                 HEADERS          ${finiteElement_headers}
                 DEPENDS_ON       ${finiteElement_dependencies}
                 )

target_include_directories( finiteElement 
                            INTERFACE 
                            $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
                            $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
                            $<INSTALL_INTERFACE:include> )

install( FILES ${finiteElement_headers} 
         DESTINATION include/finiteElement )




option(USE_SEMCLASSIC "Use SEMClassic" ON)
option(USE_SEMOPTIM "Use SEMOptim" OFF)
option(USE_SHIVA "Use SEMShiva" OFF)


# Set the appropriate values
if(USE_SEMCLASSIC)
    set(SEMQkGLIntegrals_HEADER "classic/SEMQkGLIntegralsClassic.hpp")
    set(SEMQkGLIntegrals_TYPE "classic/SEMQkGLIntegralsClassic")
elseif(USE_SEMOPTIM)
    set(SEMQkGLIntegrals_HEADER "optim/SEMQkGLIntegrals.hpp")
    set(SEMQkGLIntegrals_TYPE "optim/SEMQkGLIntegrals")
elseif(USE_SEMGEOS)
    set(SEMQkGLIntegrals_HEADER "geos/SEMQkGLIntegralsGeos.hpp")
    set(SEMQkGLIntegrals_TYPE "geos/SEMQkGLIntegralsGeos")
elseif(USE_SHIVA)
    set(SEMQkGLIntegrals_HEADER "shiva/SEMQkGLIntegralsShiva.hpp")
    set(SEMQkGLIntegrals_TYPE "shiva/SEMQkGLIntegralsShiva")
endif()

# Generate the alias header
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/SEMQkGLIntegrals.hpp.in
    ${CMAKE_CURRENT_BINARY_DIR}/SEMQkGLIntegrals.hpp
    @ONLY
    )


if( SEMKERNELS_ENABLE_UNIT_TESTS )
    add_subdirectory( unitTests )
endif()
