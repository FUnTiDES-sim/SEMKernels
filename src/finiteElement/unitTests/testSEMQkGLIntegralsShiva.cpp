
#include "../SEMQkGLIntegralsShiva.hpp"

#include <gtest/gtest.h>

using namespace shiva;
using namespace shiva::functions;
using namespace shiva::geometry;
using namespace shiva::discretizations::finiteElementMethod;

void setX( double (&X)[8][3] )
{
  double x0=0,y0=0,z0=0;
  double x1=20,y1=20,z1=20;

  double X0[8][3] = { {x0,y0,z0},
                     {x1,y0,z0},
                     {x0,y1,z0},
                     {x1,y1,z0},
                     {x0,y0,z1},
                     {x1,y0,z1},
                     {x0,y1,z1},
                     {x1,y1,z1} }; 
  
  for( int a=0; a<8; ++a )
  {
    for( int i=0; i<3; ++i )
    {
      X[a][i] *= X0[a][i] * ( 0.9 + 0.1 * (rand() % 20) );
    }
  }

}



template< typename INTEGRALS >
void runtimeCalcBKernel( double const (&X)[8][3], 
                         double (&B)[6] ) 
{
  using Integrals = INTEGRALS;
  using TransformType = typename INTEGRALS::TransformType;

  TransformType trilinearCell;
  trilinearCell.setData( X );

  pmpl::genericKernelWrapper( 6, B, [trilinearCell] SHIVA_DEVICE ( auto * device_data )
  {
    double B[6] = {0.0};
    Integrals::template computeB<0,0,0>( trilinearCell, B );
    device_data[0] = B[0];
    device_data[1] = B[1];
    device_data[2] = B[2];
    device_data[3] = B[3];
    device_data[4] = B[4];
    device_data[5] = B[5];
  } );
}

TEST( testIntegrals, test_computeStiffnessTerm )
{
  using TransformType =
  LinearTransform< double,
                   InterpolatedShape< double,
                                     Cube< double >,
                                     LagrangeBasis< double, 1, EqualSpacing >,
                                     LagrangeBasis< double, 1, EqualSpacing >,
                                     LagrangeBasis< double, 1, EqualSpacing > > >;

  using ParentElementType =
  ParentElement< double,
                 Cube< double >,
                 LagrangeBasis< double, 1, EqualSpacing >,
                 LagrangeBasis< double, 1, EqualSpacing >,
                 LagrangeBasis< double, 1, EqualSpacing > >;

  using Integrals = SEMQkGLIntegralsShiva< double, 1, TransformType, ParentElementType >;
  

  double X[8][3]; 
  double B[6] = {0};
  setX(X);

  runtimeCalcBKernel<Integrals>( X, B );
}


int main( int argc, char * * argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  int const result = RUN_ALL_TESTS();
  return result;
}
