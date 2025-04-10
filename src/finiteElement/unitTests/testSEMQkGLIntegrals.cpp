
#include "../SEMQkGLIntegralsClassic.hpp"
#include "../SEMQkGLIntegralsOptim.hpp"
#include "../SEMQkGLIntegralsShiva.hpp"

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
  // double x0 = -1.1, y0 = -0.9, z0 = -0.8;
  // double x1 =  1.2, y1 =  1.1, z1 =  0.9;

  double x0 = -1.0, y0 = -1.0, z0 = -1.0;
  double x1 =  1.0, y1 =  1.0, z1 =  1.0;

  X( 0, 0 ) = x0; Y( 0, 0 ) = y0; Z( 0, 0 ) = z0;
  X( 0, 1 ) = x1; Y( 0, 1 ) = y0; Z( 0, 1 ) = z0;
  X( 0, 2 ) = x0; Y( 0, 2 ) = y1; Z( 0, 2 ) = z0;
  X( 0, 3 ) = x1; Y( 0, 3 ) = y1; Z( 0, 3 ) = z0;
  X( 0, 4 ) = x0; Y( 0, 4 ) = y0; Z( 0, 4 ) = z1;
  X( 0, 5 ) = x1; Y( 0, 5 ) = y0; Z( 0, 5 ) = z1;
  X( 0, 6 ) = x0; Y( 0, 6 ) = y1; Z( 0, 6 ) = z1;
  X( 0, 7 ) = x1; Y( 0, 7 ) = y1; Z( 0, 7 ) = z1;
  
}










template< typename INTEGRALS >
void computeMassMatrixAndStiffnessVectorTester()
{
  using Integrals = INTEGRALS;
  static constexpr int order = INTEGRALS::order;


  CArrayNd<double, 1, 8> Xcoords;
  CArrayNd<double, 1, 8> Ycoords;
  CArrayNd<double, 1, 8> Zcoords;
  setXYZ( Xcoords, Ycoords, Zcoords );

  constexpr int massOffset = 0;
  constexpr int pOffset = (order+1)*(order+1)*(order+1);
  constexpr int YOffset = 2 * (order+1)*(order+1)*(order+1);
  constexpr int length = (order+1)*(order+1)*(order+1);
  
  constexpr int dataSize = 3 * (order+1)*(order+1)*(order+1);
  float * hostData = new float[ dataSize ];

  hostData[pOffset] = 1.0;

  pmpl::genericKernelWrapper( dataSize, hostData, [Xcoords, Ycoords, Zcoords, length] SHIVA_DEVICE ( auto * device_data )
  {
    float * const massMatrixLocal = device_data + massOffset;
    float * const pnLocal = device_data + pOffset;
    float * const Y = device_data + YOffset;

    Integrals::computeMassMatrixAndStiffnessVector( 0,
                                                    length,
                                                    Xcoords,
                                                    Ycoords,
                                                    Zcoords,
                                                    massMatrixLocal,
                                                    pnLocal,
                                                    Y );  
  } );

  float const massMatrixLocalSoln[] = {  0.004525463,   0.02262731,   0.02262731,  0.004525463,   0.02262731,    0.1131366,    0.1131366,   0.02262731,   0.02262731,    0.1131366,    0.1131366,   0.02262731,  0.004525463,   0.02262731,   0.02262731,  0.004525463,   0.02262731,    0.1131366,    0.1131366,   0.02262731,    0.1131366,    0.5656829,    0.5656829,    0.1131366,    0.1131366,    0.5656829,    0.5656829,    0.1131366,   0.02262731,    0.1131366,    0.1131366,   0.02262731,   0.02262731,    0.1131366,    0.1131366,   0.02262731,    0.1131366,    0.5656829,    0.5656829,    0.1131366,    0.1131366,    0.5656829,    0.5656829,    0.1131366,   0.02262731,    0.1131366,    0.1131366,   0.02262731,  0.004525463,   0.02262731,   0.02262731,  0.004525463,   0.02262731,    0.1131366,    0.1131366,   0.02262731,   0.02262731,    0.1131366,    0.1131366,   0.02262731,  0.004525463,   0.02262731,   0.02262731,  0.004525463 };
  float const pnLocalSoln[] = {            1,            0,            0,            0,            0,            0,            0,            0,            0,            0,            0,            0,            0,            0,            0,            0,            0,            0,            0,            0,            0,            0,            0,            0,            0,            0,            0,            0,            0,            0,            0,            0,            0,            0,            0,            0,            0,            0,            0,            0,            0,            0,            0,            0,            0,            0,            0,            0,            0,            0,            0,            0,            0,            0,            0,            0,            0,            0,            0,            0,            0,            0,            0,            0 };
  float const YSoln[] = {    0.2557976,    0.0615942,    0.0615942,    0.0615942,   0.08145833,  9.555022e-35, -3.007746e-34, -2.031604e-34,   0.08145833,  4.684745e-34,  7.214965e-35,  1.697639e-34,   0.08145833,  2.987106e-34, -9.761423e-35,            0,    0.1127451,  4.467273e-34, -1.018192e-34,  3.328704e-35, -3.278528e-34,            0,            0,            0,  3.547664e-34,            0,            0,            0,  4.402211e-35,            0,            0,            0,    0.1127451,  1.021611e-33,  4.730649e-34,  6.081711e-34,  4.324314e-34,            0,            0,            0,  1.115051e-33,            0,            0,            0,  8.043063e-34,            0,            0,            0,    0.1127451,  4.134403e-34, -1.351062e-34,            0, -3.718749e-34,            0,            0,            0,  3.107443e-34,            0,            0,            0,            0,            0,            0,            0 };

  float maxMassSoln = 0.0;
  float maxPNLocalSoln = 0.0;
  float maxYSoln = 0.0;
  for( int i = 0; i < length; ++i )
  {
    maxMassSoln = std::max( maxMassSoln, std::abs( massMatrixLocalSoln[i] ) );
    maxPNLocalSoln = std::max( maxPNLocalSoln, std::abs( pnLocalSoln[i] ) );
    maxYSoln = std::max( maxYSoln, std::abs( YSoln[i] ) );
  }
  
  for( int i = 0; i < length; ++i )
  {
//    EXPECT_NEAR( massMatrixLocalSoln[i], hostData[massOffset + i], maxMassSoln * 1e-4 );
    // EXPECT_NEAR( pnLocalSoln[i], hostData[pOffset + i], maxPNLocalSoln * 1e-4 );
    // EXPECT_NEAR( YSoln[i], hostData[YOffset + i], maxYSoln * 1e-4 );
  }
}

TEST( testSEMQkGLIntegralsShiva, computeMassMatrixAndStiffnessVector )
{
  using TransformType =
    LinearTransform< double,
                     InterpolatedShape< double,
                                        Cube< double >,
                                        LagrangeBasis< double, 1, EqualSpacing >,
                                        LagrangeBasis< double, 1, EqualSpacing >,
                                        LagrangeBasis< double, 1, EqualSpacing > > >;

  constexpr int order = 2;
  using ParentElementType =
    ParentElement< double,
                   Cube< double >,
                   LagrangeBasis< double, order, EqualSpacing >,
                   LagrangeBasis< double, order, EqualSpacing >,
                   LagrangeBasis< double, order, EqualSpacing > >;

  using Integrals = SEMQkGLIntegralsShiva< double, order, TransformType, ParentElementType >;

  computeMassMatrixAndStiffnessVectorTester< Integrals >();
}


TEST( testSEMQkGLIntegralsOptim, computeMassMatrixAndStiffnessVector )
{
  constexpr int order = 2;
  using Integrals = SEMQkGLIntegralsOptim<order>;

  computeMassMatrixAndStiffnessVectorTester< Integrals >();
}

TEST( testSEMQkGLIntegralsClassic, computeMassMatrixAndStiffnessVector )
{
  constexpr int order = 1;
  using Integrals = SEMQkGLIntegralsClassic<order>;

  computeMassMatrixAndStiffnessVectorTester< Integrals >();
}

int main( int argc, char * * argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  int const result = RUN_ALL_TESTS();
  return result;
}
