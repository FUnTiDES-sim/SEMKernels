#pragma once

#include "common/macros.hpp"

#include "functions/bases/LagrangeBasis.hpp"
#include "functions/quadrature/Quadrature.hpp"
#include "functions/spacing/Spacing.hpp"
#include "geometry/shapes/NCube.hpp"
#include "geometry/shapes/InterpolatedShape.hpp"
#include "geometry/mapping/LinearTransform.hpp"
#include "common/ShivaMacros.hpp"
#include "common/pmpl.hpp"
#include "common/types.hpp"
#include "discretizations/finiteElementMethod/parentElements/ParentElement.hpp"

using namespace shiva;
using namespace shiva::functions;
using namespace shiva::geometry;
using namespace shiva::geometry::utilities;
using namespace shiva::discretizations::finiteElementMethod;

using namespace std;

/**
 * This class is the basis class for the hexahedron finite element cells with shape functions defined on Gauss-Lobatto quadrature points.
 */
template< typename REAL_TYPE, 
          int ORDER,
          typename TRANSFORM, 
          typename PARENT_ELEMENT >
class SEMQkGLIntegralsShiva
{
public:

  static constexpr int order = ORDER;
  static constexpr int numPoints = ORDER + 1;

  using TransformType = TRANSFORM;

  using ParentElementType = PARENT_ELEMENT;

  using JacobianType = typename std::remove_reference_t< TransformType >::JacobianType;
  using quadrature = QuadratureGaussLobatto<double, numPoints >;
  using basisFunction=LagrangeBasis< double, ORDER, GaussLobattoSpacing >;

  SEMKERNELS_HOST_DEVICE SEMQkGLIntegralsShiva(){}
  SEMKERNELS_HOST_DEVICE ~SEMQkGLIntegralsShiva(){}
  
  /////////////////////////////////////////////////////////////////////////////////////
  //  from GEOS implementation
  /////////////////////////////////////////////////////////////////////////////////////

  /**
   * @brief Calculates the linear index for support/quadrature points from ijk
   *   coordinates.
   * @param r order of polynomial approximation
   * @param i The index in the xi0 direction (0,r)
   * @param j The index in the xi1 direction (0,r)
   * @param k The index in the xi2 direction (0,r)
   * @return The linear index of the support/quadrature point (0-(r+1)^3)
   */
  static constexpr inline
  SEMKERNELS_HOST_DEVICE 
  int linearIndex( const int r,
                   const int i,
                   const int j,
                   const int k ) 
  {
    return i + (r+1) * j + (r+1)*(r+1) * k;
  }


  /**
   * @brief Invert the symmetric matrix @p srcSymMatrix and store the result in @p dstSymMatrix.
   * @param dstSymMatrix The 3x3 symmetric matrix to write the inverse to.
   * @param srcSymMatrix The 3x3 symmetric matrix to take the inverse of.
   * @return The determinant.
   * @note @p srcSymMatrix can contain integers but @p dstMatrix must contain floating point values.
   */
  static constexpr inline
  SEMKERNELS_HOST_DEVICE 
  void symInvert( double  dstSymMatrix[6], double  srcSymMatrix[6]) 
  {
   
     using FloatingPoint = std::decay_t< decltype( dstSymMatrix[ 0 ] ) >;
   
     dstSymMatrix[ 0 ] = srcSymMatrix[ 1 ] * srcSymMatrix[ 2 ] - srcSymMatrix[ 3 ] * srcSymMatrix[ 3 ];
     dstSymMatrix[ 5 ] = srcSymMatrix[ 4 ] * srcSymMatrix[ 3 ] - srcSymMatrix[ 5 ] * srcSymMatrix[ 2 ];
     dstSymMatrix[ 4 ] = srcSymMatrix[ 5 ] * srcSymMatrix[ 3 ] - srcSymMatrix[ 4 ] * srcSymMatrix[ 1 ];
   
     double det = srcSymMatrix[ 0 ] * dstSymMatrix[ 0 ] + srcSymMatrix[ 5 ] * dstSymMatrix[ 5 ] + srcSymMatrix[ 4 ] * dstSymMatrix[ 4 ];
  
     FloatingPoint const invDet = FloatingPoint( 1 ) / det;
   
     dstSymMatrix[ 0 ] *= invDet;
     dstSymMatrix[ 5 ] *= invDet;
     dstSymMatrix[ 4 ] *= invDet;
     dstSymMatrix[ 1 ] = ( srcSymMatrix[ 0 ] * srcSymMatrix[ 2 ] - srcSymMatrix[ 4 ] * srcSymMatrix[ 4 ] ) * invDet;
     dstSymMatrix[ 3 ] = ( srcSymMatrix[ 5 ] * srcSymMatrix[ 4 ] - srcSymMatrix[ 0 ] * srcSymMatrix[ 3 ] ) * invDet;
     dstSymMatrix[ 2 ] = ( srcSymMatrix[ 0 ] * srcSymMatrix[ 1 ] - srcSymMatrix[ 5 ] * srcSymMatrix[ 5 ] ) * invDet;
   
  }
  
  /**
   * @brief Invert the symmetric matrix @p symMatrix overwritting it.
   * @param symMatrix The 3x3 symmetric matrix to take the inverse of and overwrite.
   * @return The determinant.
   * @note @p symMatrix can contain integers but @p dstMatrix must contain floating point values.
  */
  static inline
  SEMKERNELS_HOST_DEVICE  
  void symInvert0( double  symMatrix[6] )
  {
      std::remove_reference_t< decltype( symMatrix[ 0 ] ) > temp[ 6 ];
      symInvert( temp, symMatrix );
      
      symMatrix[0]=temp[0];
      symMatrix[1]=temp[1];
      symMatrix[2]=temp[2];
      symMatrix[3]=temp[3];
      symMatrix[4]=temp[4];
      symMatrix[5]=temp[5];
  }
  







  template< int qa, int qb,  int qc, typename FUNC>
  static constexpr inline
  SEMKERNELS_HOST_DEVICE
  void computeGradPhiBGradPhi( double const (&B)[6],
                               FUNC && func )
  {
     constexpr double qcoords[3] = { quadrature::template coordinate<qa>(),
                                     quadrature::template coordinate<qb>(),
                                     quadrature::template coordinate<qc>() };
     constexpr double w = quadrature::template weight<qa>() * quadrature::template weight<qb>() * quadrature::template weight<qc>();
     for( int i=0; i<ORDER+1; i++ )
     {
       const int ibc = linearIndex( ORDER,  i, qb, qc );
       const int aic = linearIndex( ORDER, qa,  i, qc );
       const int abi = linearIndex( ORDER, qa, qb,  i );
       const double gia = basisFunction::template gradient<qa>(qcoords[0]);
       const double gib = basisFunction::template gradient<qb>(qcoords[1]);
       const double gic = basisFunction::template gradient<qc>(qcoords[2]);
       for( int j=0; j<ORDER+1; j++ )
       {
         const int jbc = linearIndex( ORDER,  j, qb, qc );
         const int ajc = linearIndex( ORDER, qa,  j, qc );
         const int abj = linearIndex( ORDER, qa, qb,  j );
         const double gja = basisFunction::template gradient<qa>(qcoords[0]);
         const double gjb = basisFunction::template gradient<qb>(qcoords[1]);
         const double gjc = basisFunction::template gradient<qc>(qcoords[2]);
         // diagonal terms
         const double w0 = w * gia * gja;
         func( ibc, jbc, w0 * B[0] );
         const double w1 = w * gib * gjb;
         func( aic, ajc, w1 * B[1] );
         const double w2 = w * gic * gjc;
         func( abi, abj, w2 * B[2] );
         // off-diagonal terms
         const double w3 = w * gib * gjc;
         func( aic, abj, w3 * B[3] );
         func( abj, aic, w3 * B[3] );
         const double w4 = w * gia * gjc;
         func( ibc, abj, w4 * B[4] );
         func( abj, ibc, w4 * B[4] );
         const double w5 = w * gia * gjb;
         func( ibc, ajc, w5 * B[5] );
         func( ajc, ibc, w5 * B[5] );
       }
     }
  }

  template< int qa, int qb, int qc >
  static constexpr inline
  SEMKERNELS_HOST_DEVICE
  void computeB( TransformType const & trilinearCell,
                 double (&B)[6] )
  {
    JacobianType J{ 0.0 };   
    shiva::geometry::utilities::jacobian<quadrature, qa,qb,qc>( trilinearCell, J );
     
    // detJ
    double const detJ = +J(0,0)*(J(1,1)*J(2,2)-J(2,1)*J(1,2))
                        -J(0,1)*(J(1,0)*J(2,2)-J(2,0)*J(1,2))
                        +J(0,2)*(J(1,0)*J(2,1)-J(2,0)*J(1,1));
    
    // compute J^{T}J/detJ
    double const invDetJ = 1.0 / detJ;
    B[0] = (J(0,0)*J(0,0)+J(1,0)*J(1,0)+J(2,0)*J(2,0)) * invDetJ;
    B[1] = (J(0,1)*J(0,1)+J(1,1)*J(1,1)+J(2,1)*J(2,1)) * invDetJ;
    B[2] = (J(0,2)*J(0,2)+J(1,2)*J(1,2)+J(2,2)*J(2,2)) * invDetJ;
    B[3] = (J(0,1)*J(0,2)+J(1,1)*J(1,2)+J(2,1)*J(2,2)) * invDetJ;
    B[4] = (J(0,0)*J(0,2)+J(1,0)*J(1,2)+J(2,0)*J(2,2)) * invDetJ;
    B[5] = (J(0,0)*J(0,1)+J(1,0)*J(1,1)+J(2,0)*J(2,1)) * invDetJ;
    // compute detJ*J^{-1}J^{-T}
    symInvert0( B );
  }

  template< typename FUNC>
  static constexpr inline
  SEMKERNELS_HOST_DEVICE
  void computeStiffnessTerm( TransformType const & trilinearCell,
                             float massMatrix[],
                             FUNC && func )
  {
        double B[6] = {0};
        JacobianType J{ 0.0 };

        // this is a compile time quadrature loop over each tensor direction
        forNestedSequence< ORDER+1,
                           ORDER+1,
                           ORDER+1 >( [&]( auto const icqa,
                                           auto const icqb,
                                           auto const icqc )
        {
          constexpr int qa = decltype(icqa)::value;
          constexpr int qb = decltype(icqb)::value;
          constexpr int qc = decltype(icqc)::value;

          // must be here, Jacobian must be put to 0 for each quadrature point
          //Jacobian matrix J
          
          J(0,0)=0;
          J(0,1)=0;
          J(0,2)=0;
          J(1,0)=0;
          J(1,1)=0;
          J(1,2)=0;
          J(2,0)=0;
          J(2,1)=0;
          J(2,2)=0;
         
          shiva::geometry::utilities::jacobian<quadrature, qa,qb,qc>( trilinearCell, J );
           
          // detJ
          // j(0,0) j(0,1) j(0,2)
          // j(1,0) j(1,1) j(1,2)
          // j(2,0) j(2,1) j(2,2)
          double const detJ = +J(0,0)*(J(1,1)*J(2,2)-J(2,1)*J(1,2))
                              -J(0,1)*(J(1,0)*J(2,2)-J(2,0)*J(1,2))
                              +J(0,2)*(J(1,0)*J(2,1)-J(2,0)*J(1,1));
          
          // mass matrix
          constexpr int q=qc+qb*(ORDER+1)+qa*(ORDER+1)*(ORDER+1);
          constexpr double w3D = quadrature::template weight<qa>()*
                                 quadrature::template weight<qb>()*
                                 quadrature::template weight<qc>();
          massMatrix[q]=w3D*detJ;

          // compute J^{T}J/detJ
          double const invDetJ = 1.0 / detJ;
          B[0] = (J(0,0)*J(0,0)+J(1,0)*J(1,0)+J(2,0)*J(2,0)) * invDetJ;
          B[1] = (J(0,1)*J(0,1)+J(1,1)*J(1,1)+J(2,1)*J(2,1)) * invDetJ;
          B[2] = (J(0,2)*J(0,2)+J(1,2)*J(1,2)+J(2,2)*J(2,2)) * invDetJ;
          B[3] = (J(0,1)*J(0,2)+J(1,1)*J(1,2)+J(2,1)*J(2,2)) * invDetJ;
          B[4] = (J(0,0)*J(0,2)+J(1,0)*J(1,2)+J(2,0)*J(2,2)) * invDetJ;
          B[5] = (J(0,0)*J(0,1)+J(1,0)*J(1,1)+J(2,0)*J(2,1)) * invDetJ;
          // compute detJ*J^{-1}J^{-T}
          symInvert0( B );

          // compute gradPhiI*B*gradPhiJ and stiffness vector
          computeGradPhiBGradPhi<ORDER,qa,qb,qc>(B, func);
      });
  }


  template< typename ARRAY_REAL_VIEW, typename LOCAL_ARRAY >
  static constexpr inline
  SEMKERNELS_HOST_DEVICE
  void
  gatherCoordinates( const int & elementNumber,
                     ARRAY_REAL_VIEW const & nodesCoordsX,
                     ARRAY_REAL_VIEW const & nodesCoordsY,
                     ARRAY_REAL_VIEW const & nodesCoordsZ,
                     LOCAL_ARRAY & cellData )
  {
      for( int k=0; k<2; ++k)
      {
          for ( int j=0; j<2; ++j)
          {
              for( int i=0; i<2; ++i)
              {
                  int const l = ORDER * ( i + j*(ORDER+1) + k*(ORDER+1)*(ORDER+1) ) ;
                  cellData(i,j,k,0) = nodesCoordsX(elementNumber,l);
                  cellData(i,j,k,1) = nodesCoordsZ(elementNumber,l);
                  cellData(i,j,k,2) = nodesCoordsY(elementNumber,l);
              }
          }
      }
  }

  /**
   * @brief compute  mass Matrix stiffnessVector.
   */
  template< typename ARRAY_REAL_VIEW > 
  static constexpr inline
  SEMKERNELS_HOST_DEVICE 
  void computeMassMatrixAndStiffnessVector(const int & elementNumber,
                                           const int & nPointsPerElement,
                                           ARRAY_REAL_VIEW const & nodesCoordsX,
                                           ARRAY_REAL_VIEW const & nodesCoordsY,
                                           ARRAY_REAL_VIEW const & nodesCoordsZ,
                                           float massMatrixLocal[],
                                           float pnLocal[],
                                           float Y[])
  {
      TransformType trilinearCell;
      typename TransformType::DataType & cellCoordData = trilinearCell.getData();

      for( int k=0; k<2; ++k)
      {
          for ( int j=0; j<2; ++j)
          {
              for( int i=0; i<2; ++i)
              {
                  int const l = ORDER * ( i + j*(ORDER+1) + k*(ORDER+1)*(ORDER+1) ) ;
                  cellCoordData(i,j,k,0) = nodesCoordsX(elementNumber,l);
                  cellCoordData(i,j,k,1) = nodesCoordsZ(elementNumber,l);
                  cellCoordData(i,j,k,2) = nodesCoordsY(elementNumber,l);
              }
          }
      }
      for (int q=0;q<nPointsPerElement;q++)
      {
         Y[q]=0;
      }
      computeStiffnessTerm<ORDER>(elementNumber,trilinearCell, massMatrixLocal, [&] (const int i, const int j, const double val)
                                 {
                                    Y[i]= Y[i] + val*pnLocal[j];
                                 });
  }
  /////////////////////////////////////////////////////////////////////////////////////
  //  end from GEOS implementation
  /////////////////////////////////////////////////////////////////////////////////////
  
};
  