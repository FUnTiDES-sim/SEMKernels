
#include "../SEMQkGLIntegralsShiva.hpp"

#include <gtest/gtest.h>

using namespace shiva;
using namespace shiva::functions;
using namespace shiva::geometry;
using namespace shiva::discretizations::finiteElementMethod;

// void test_calculateB()
// {

// }


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
  

  double x0=0,y0=0,z0=0;
  double x1=20,y1=20,z1=20;
  double X[8][3]={{x0,y0,z0},{x1,y0,z0},{x0,y1,z0},{x1,y1,z0},
                  {x0,y0,z1},{x1,y0,z1},{x0,y1,z1},{x1,y1,z1}}; 


  TransformType trilinearCell;
  trilinearCell.setData( X );
  double B[6] = {0.0};

  Integrals::computeB<0,0,0>( trilinearCell, B );

}


int main( int argc, char * * argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  int const result = RUN_ALL_TESTS();
  return result;
}
