#include "SEMdata.hpp"

using tfloat = float;
using gfloat = float;

#ifdef USE_SEMCLASSIC
    #include <fe/SEMKernels/src/finiteElement/classic/SEMQkGLBasisFunctionsClassic.hpp>
    using SEMQkGLBasisFunctions = SEMQkGLBasisFunctionsClassic<SEMinfo::myOrderNumber>;
#endif

#ifdef  USE_SEMOPTIM 
    #include <fe/SEMKernels/src/finiteElement/optim/SEMQkGLBasisFunctionsOptim.hpp>
#endif

#ifdef  USE_SEMGEOS
    #include <fe/SEMKernels/src/finiteElement/geos/SEMQkGLBasisFunctionsGeos.hpp>
#endif