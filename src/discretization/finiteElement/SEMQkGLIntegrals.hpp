#include "SEMdata.hpp"

using tfloat = float;
using gfloat = float;
using real_t = float;

#ifdef USE_SEMCLASSIC
    #include <fe/SEMKernels/src/discretization/finiteElement/classic/SEMQkGLIntegralsClassic.hpp>
    using SEMQkGLIntegrals = SEMQkGLIntegralsClassic ;
#endif
#ifdef  USE_SEMOPTIM 
    #include <fe/SEMKernels/src/discretization/finiteElement/optim/SEMQkGLIntegralsOptim.hpp>
    using SEMQkGLIntegrals = SEMQkGLIntegralsOptim<SEMinfo::myOrderNumber, tfloat, gfloat>;
#endif
#ifdef  USE_SEMGEOS 
    #include <fe/SEMKernels/src/discretization/finiteElement/geos/SEMQkGLIntegralsGeos.hpp>
    //using SEMQkGLIntegrals = SEMQkGLIntegralsGeos<SEMinfo::myOrderNumber, tfloat, gfloat>;
    using SEMQkGLIntegrals = Q2_Hexahedron_Lagrange_GaussLobatto; 
#endif // USE_SEMGEOS

#ifdef USE_SHIVA
    #include <fe/SEMKernels/src/discretization/finiteElement/shiva/SEMQkGLIntegralsShiva.hpp>
    using TransformType =
    LinearTransform< tfloat,
                     InterpolatedShape< tfloat,
                                        Cube< tfloat >,
                                        LagrangeBasis< tfloat, 1, EqualSpacing >,
                                        LagrangeBasis< tfloat, 1, EqualSpacing >,
                                        LagrangeBasis< tfloat, 1, EqualSpacing > > >;

  using ParentElementType =
    ParentElement< gfloat,
                   Cube< gfloat >,
                   LagrangeBasis< gfloat, SEMinfo::myOrderNumber, EqualSpacing >,
                   LagrangeBasis< gfloat, SEMinfo::myOrderNumber, EqualSpacing >,
                   LagrangeBasis< gfloat, SEMinfo::myOrderNumber, EqualSpacing > >;

  using SEMQkGLIntegrals = SEMQkGLIntegralsShiva< SEMinfo::myOrderNumber, TransformType, ParentElementType >;    
#endif
