
#include "../SEMQkGLIntegralsShiva.hpp"
#include "../SEMQkGLIntegralsOptim.hpp"

#include <gtest/gtest.h>

using namespace shiva;
using namespace shiva::functions;
using namespace shiva::geometry;
using namespace shiva::discretizations::finiteElementMethod;

void setX( double (&X)[8][3] )
{
  double x0 = 0, y0 = 0, z0 = 0;
  double x1 = 20, y1 = 20, z1 = 20;

  double X0[8][3] = { {x0, y0, z0},
    {x1, y0, z0},
    {x0, y1, z0},
    {x1, y1, z0},
    {x0, y0, z1},
    {x1, y0, z1},
    {x0, y1, z1},
    {x1, y1, z1} };

  for ( int a = 0; a < 8; ++a )
  {
    for ( int i = 0; i < 3; ++i )
    {
      X[a][i] = X0[a][i] ;//* ( 0.9 + 0.1 * (rand() % 20) );
    }
  }
}

template< typename T >
void setXYZ( T & X, T & Y, T & Z )
{
  double x0 = -1.1, y0 = -0.9, z0 = -0.8;
  double x1 =  1.2, y1 =  1.1, z1 =  0.9;

  X( 0, 0 ) = x0; Y( 0, 0 ) = y0; Z( 0, 0 ) = z0;
  X( 0, 1 ) = x1; Y( 0, 1 ) = y0; Z( 0, 1 ) = z0;
  X( 0, 2 ) = x0; Y( 0, 2 ) = y1; Z( 0, 2 ) = z0;
  X( 0, 3 ) = x1; Y( 0, 3 ) = y1; Z( 0, 3 ) = z0;
  X( 0, 4 ) = x0; Y( 0, 4 ) = y0; Z( 0, 4 ) = z1;
  X( 0, 5 ) = x1; Y( 0, 5 ) = y0; Z( 0, 5 ) = z1;
  X( 0, 6 ) = x0; Y( 0, 6 ) = y1; Z( 0, 6 ) = z1;
  X( 0, 7 ) = x1; Y( 0, 7 ) = y1; Z( 0, 7 ) = z1;
  
}


// template< typename INTEGRALS >
// void runtimeCalcBKernel( double const (&X)[8][3],
//                          double (&B)[6] )
// {
//   using Integrals = INTEGRALS;
//   using TransformType = typename INTEGRALS::TransformType;

//   TransformType trilinearCell;
//   trilinearCell.setData( X );

//   pmpl::genericKernelWrapper( 6, B, [trilinearCell] SHIVA_DEVICE ( auto * device_data )
//   {
//     double B[6] = {0.0};
//     Integrals::template computeB< 0, 0, 0 >( trilinearCell, B );
//     device_data[0] = B[0];
//     device_data[1] = B[1];
//     device_data[2] = B[2];
//     device_data[3] = B[3];
//     device_data[4] = B[4];
//     device_data[5] = B[5];
//   } );
// }

// TEST( testIntegrals, test_computeStiffnessTerm )
// {
//   using TransformType =
//     LinearTransform< double,
//                      InterpolatedShape< double,
//                                         Cube< double >,
//                                         LagrangeBasis< double, 1, EqualSpacing >,
//                                         LagrangeBasis< double, 1, EqualSpacing >,
//                                         LagrangeBasis< double, 1, EqualSpacing > > >;

//   using ParentElementType =
//     ParentElement< double,
//                    Cube< double >,
//                    LagrangeBasis< double, 3, EqualSpacing >,
//                    LagrangeBasis< double, 3, EqualSpacing >,
//                    LagrangeBasis< double, 3, EqualSpacing > >;

//   using Integrals = SEMQkGLIntegralsShiva< double, 3, TransformType, ParentElementType >;


//   double X[8][3];
//   double B[6] = {0};
//   setX( X );

//   runtimeCalcBKernel< Integrals >( X, B );
// }








// template< typename INTEGRALS >
// void runtimeCalcBKernel( double const (&X)[8][3],
//                          double (&B)[6] )
// {
//   using Integrals = INTEGRALS;
//   using TransformType = typename INTEGRALS::TransformType;

//   TransformType trilinearCell;
//   trilinearCell.setData( X );

//   pmpl::genericKernelWrapper( 6, B, [trilinearCell] SHIVA_DEVICE ( auto * device_data )
//   {
//     double B[6] = {0.0};
//     Integrals::template computeB< 0, 0, 0 >( trilinearCell, B );
//     device_data[0] = B[0];
//     device_data[1] = B[1];
//     device_data[2] = B[2];
//     device_data[3] = B[3];
//     device_data[4] = B[4];
//     device_data[5] = B[5];
//   } );
// }

TEST( testSEMQkGLIntegralsShiva, computeMassMatrixAndStiffnessVector )
{
  using TransformType =
    LinearTransform< double,
                     InterpolatedShape< double,
                                        Cube< double >,
                                        LagrangeBasis< double, 1, EqualSpacing >,
                                        LagrangeBasis< double, 1, EqualSpacing >,
                                        LagrangeBasis< double, 1, EqualSpacing > > >;

  constexpr int order = 3;                                        
  using ParentElementType =
    ParentElement< double,
                   Cube< double >,
                   LagrangeBasis< double, order, EqualSpacing >,
                   LagrangeBasis< double, order, EqualSpacing >,
                   LagrangeBasis< double, order, EqualSpacing > >;

  using Integrals = SEMQkGLIntegralsShiva< double, order, TransformType, ParentElementType >;


  CArrayNd<double, 1, 8> Xcoords;
  CArrayNd<double, 1, 8> Ycoords;
  CArrayNd<double, 1, 8> Zcoords;
  setXYZ( Xcoords, Ycoords, Zcoords );

  float massMatrixLocal[ (order+1)*(order+1)*(order+1) ] = {0};
  float pnLocal[(order+1)*(order+1)*(order+1)] = {0};
  float Y[(order+1)*(order+1)*(order+1)] = {0};
  
  pnLocal[0] = 1.0;
  
  Integrals::computeMassMatrixAndStiffnessVector( 0,
                                                  8,
                                                  Xcoords,
                                                  Ycoords,
                                                  Zcoords,
                                                  massMatrixLocal,
                                                  pnLocal,
                                                  Y );
}


TEST( testSEMQkGLIntegralsOptim, computeMassMatrixAndStiffnessVector )
{
  using Integrals = SEMQkGLIntegralsOptim;
  Integrals integrals;


  CArrayNd<double, 1, 8> Xcoords;
  CArrayNd<double, 1, 8> Ycoords;
  CArrayNd<double, 1, 8> Zcoords;
  setXYZ( Xcoords, Ycoords, Zcoords );

  float massMatrixLocal[ (order+1)*(order+1)*(order+1) ] = {0};
  float pnLocal[(order+1)*(order+1)*(order+1)] = {0};
  float Y[(order+1)*(order+1)*(order+1)] = {0};
  
  pnLocal[0] = 1.0;
  
  Integrals::computeMassMatrixAndStiffnessVector( 0,
                                                  8,
                                                  Xcoords,
                                                  Ycoords,
                                                  Zcoords,
                                                  massMatrixLocal,
                                                  pnLocal,
                                                  Y );
}

int main( int argc, char * * argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  int const result = RUN_ALL_TESTS();
  return result;
}
