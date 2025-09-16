#ifndef SEMQKGLINTEGRALSCLASSIC_HPP_
#define SEMQKGLINTEGRALSCLASSIC_HPP_

#include <data_type.h>
#include <sem_macros.h>
#include <fe/SEMKernels/src/finiteElement/classic/SEMQkGLBasisFunctionsClassic.hpp>

using namespace std;

/**
 * This class is the basis class for the hexahedron finite element cells with shape functions defined on Gauss-Lobatto quadrature points.
 */
template< int ORDER >
class SEMQkGLIntegralsClassic
{
private:
  SEMQkGLBasisFunctionsClassic GLBasis;

public:
  constexpr static bool isClassic = true;

  PROXY_HOST_DEVICE SEMQkGLIntegralsClassic(){};
  PROXY_HOST_DEVICE ~SEMQkGLIntegralsClassic(){};

  struct PrecomputedData
  {
    float quadraturePoints[ ORDER + 1 ];
    float weights[ ORDER + 1 ];
    float derivativeBasisFunction1D[ORDER + 1][ ORDER + 1 ];
  };

  PROXY_HOST_DEVICE
  static void init( PrecomputedData & precomputedData )
  {
    // initialize quadrature points and weights
    SEMQkGLBasisFunctionsClassic::gaussLobattoQuadraturePoints( ORDER, precomputedData.quadraturePoints );

    SEMQkGLBasisFunctionsClassic::gaussLobattoQuadratureWeights( ORDER, precomputedData.weights );

    // initialize derivative basis function
    SEMQkGLBasisFunctionsClassic::getDerivativeBasisFunction1D( ORDER,
                                                                precomputedData.quadraturePoints,
                                                                precomputedData.derivativeBasisFunction1D );
  }

  // compute B and M
  PROXY_HOST_DEVICE static void computeB( const int & elementNumber,
                                     const int & order,
                                     float const (&weights)[ORDER + 1],
                                     ARRAY_REAL_VIEW const & nodesCoordsX,
                                     ARRAY_REAL_VIEW const & nodesCoordsY,
                                     ARRAY_REAL_VIEW const & nodesCoordsZ,
                                     float const (&dPhi)[ORDER + 1][ORDER + 1],
                                     float massMatrixLocal[],
                                     float B[][COL] )
  {
      for( int i3=0; i3<order+1; i3++ )
      {
        for( int i2=0; i2<order+1; i2++ )
        {
          for( int i1=0; i1<order+1; i1++ )
          {
            int i=i1+i2*(order+1)+i3*(order+1)*(order+1);
            // compute jacobian matrix
            float jac00=0;
            float jac01=0;
            float jac02=0;
            float jac10=0;
            float jac11=0;
            float jac12=0;
            float jac20=0;
            float jac21=0;
            float jac22=0;
   
            for( int j1=0; j1<order+1; j1++ )
            {
              int j=j1+i2*(order+1)+i3*(order+1)*(order+1);
              float X=nodesCoordsX( elementNumber, j );
              float Y=nodesCoordsY( elementNumber, j );
              float Z=nodesCoordsZ( elementNumber, j );
              jac00+=X*dPhi[ i1 ][ j1 ];
              jac20+=Y*dPhi[ i1 ][ j1 ];
              jac10+=Z*dPhi[ i1 ][ j1 ];
              }
            for( int j2=0; j2<order+1; j2++ )
            {
              int j=i1+j2*(order+1)+i3*(order+1)*(order+1);
              float X=nodesCoordsX( elementNumber, j );
              float Y=nodesCoordsY( elementNumber, j );
              float Z=nodesCoordsZ( elementNumber, j );
              jac01+=X*dPhi[ i2 ][ j2 ];
              jac21+=Y*dPhi[ i2 ][ j2 ];
              jac11+=Z*dPhi[ i2 ][ j2 ];
            }
            for( int j3=0; j3<order+1; j3++ )
            {
              int j=i1+i2*(order+1)+j3*(order+1)*(order+1);
              float X=nodesCoordsX( elementNumber, j );
              float Y=nodesCoordsY( elementNumber, j );
              float Z=nodesCoordsZ( elementNumber, j );
              jac02+=X*dPhi[ i3 ][ j3 ];
              jac22+=Y*dPhi[ i3 ][ j3 ];
              jac12+=Z*dPhi[ i3 ][ j3 ];
            }
            // detJ
            float detJ=abs( jac00*(jac11*jac22-jac21*jac12)
                             -jac01*(jac10*jac22-jac20*jac12)
                             +jac02*(jac10*jac21-jac20*jac11));
   
            // inv of jac is equal of the minors of the transposed of jac
            float invJac00=jac11*jac22-jac12*jac21;
            float invJac01=jac02*jac21-jac01*jac22;
            float invJac02=jac01*jac12-jac02*jac11;
            float invJac10=jac12*jac20-jac10*jac22;
            float invJac11=jac00*jac22-jac02*jac20;
            float invJac12=jac02*jac10-jac00*jac12;
            float invJac20=jac10*jac21-jac11*jac20;
            float invJac21=jac01*jac20-jac00*jac21;
            float invJac22=jac00*jac11-jac01*jac10;
    
            float transpInvJac00=invJac00;
            float transpInvJac01=invJac10;
            float transpInvJac02=invJac20;
            float transpInvJac10=invJac01;
            float transpInvJac11=invJac11;
            float transpInvJac12=invJac21;
            float transpInvJac20=invJac02;
            float transpInvJac21=invJac12;
            float transpInvJac22=invJac22;
    
            float detJM1=1./detJ;
    
            // B
            B[i][0]=(invJac00*transpInvJac00+invJac01*transpInvJac10+invJac02*transpInvJac20)*detJM1;    //B11
            B[i][1]=(invJac10*transpInvJac01+invJac11*transpInvJac11+invJac12*transpInvJac21)*detJM1;    //B22
            B[i][2]=(invJac20*transpInvJac02+invJac21*transpInvJac12+invJac22*transpInvJac22)*detJM1;    //B33
            B[i][3]=(invJac00*transpInvJac01+invJac01*transpInvJac11+invJac02*transpInvJac21)*detJM1;    //B12,B21
            B[i][4]=(invJac00*transpInvJac02+invJac01*transpInvJac12+invJac02*transpInvJac22)*detJM1;    //B13,B31
            B[i][5]=(invJac10*transpInvJac02+invJac11*transpInvJac12+invJac12*transpInvJac22)*detJM1;    //B23,B32
    
            //M
            massMatrixLocal[i]=weights[i1]*weights[i2]*weights[i3]*detJ;
          }
         }
       }
  }
    
  // compute the matrix $R_{i,j}=\int_{K}{\nabla{\phi_i}.\nabla{\phi_j}dx}$
  // Marc Durufle Formulae
  PROXY_HOST_DEVICE 
  static void gradPhiGradPhi( const int & nPointsPerElement,
                              const int & order,
                              float const (&weights)[ORDER + 1],
                              float const (&dPhi)[ORDER + 1][ORDER + 1],
                              float const B[][COL],
                              float const pnLocal[],
                              float R[],
                              float Y[] )
  {
      int orderPow2=(order+1)*(order+1);
      for( int i3=0; i3<order+1; i3++ )
      {
        for( int i2=0; i2<order+1; i2++ )
        {
          for( int i1=0; i1<order+1; i1++ )
          {
            for( int j=0; j<nPointsPerElement; j++ )
            {
              R[j]=0;
            }
   
            //B11
            for( int j1=0; j1<order+1; j1++ )
            {
              int j=j1+i2*(order+1)+i3*orderPow2;
              for( int l=0; l<order+1; l++ )
              {
                int ll=l+i2*(order+1)+i3*orderPow2;
                R[j]+=weights[l]*weights[i2]*weights[i3]*(B[ll][0]*dPhi[ l ][ i1 ]*dPhi[ l ][ j1 ]);
              }
            }
            //B22
            for( int j2=0; j2<order+1; j2++ )
            {
              int j=i1+j2*(order+1)+i3*orderPow2;
              for( int m=0; m<order+1; m++ )
              {
                int mm=i1+m*(order+1)+i3*orderPow2;
                R[j]+=weights[i1]*weights[m]*weights[i3]*(B[mm][1]*dPhi[ m ][ i2 ]*dPhi[ m ][ j2 ]);
              }
            }
            //B33
            for( int j3=0; j3<order+1; j3++ )
            {
              int j=i1+i2*(order+1)+j3*orderPow2;
              for( int n=0; n<order+1; n++ )
              {
                int nn=i1+i2*(order+1)+n*orderPow2;
                R[j]+=weights[i1]*weights[i2]*weights[n]*(B[nn][2]*dPhi[ n ][ i3 ]*dPhi[ n ][ j3 ]);
              }
            }
            // B12,B21 (B[][3])
            for( int j2=0; j2<order+1; j2++ )
            {
              for( int j1=0; j1<order+1; j1++ )
              {
                int j=j1+j2*(order+1)+i3*orderPow2;
                int k=j1+i2*(order+1)+i3*orderPow2;
                int l=i1+j2*(order+1)+i3*orderPow2;
                R[j]+= weights[j1]*weights[i2]*weights[i3]*( B[k][3]*dPhi[ j1 ][ i1 ]*dPhi[ i2 ][ j2 ] ) +
                       weights[i1]*weights[j2]*weights[i3]*( B[l][3]*dPhi[ i1 ][ j1 ]*dPhi[ j2 ][ i2 ] ) ;
              }
            }
            // B13,B31 (B[][4])
            for( int j3=0; j3<order+1; j3++ )
            {
              for( int j1=0; j1<order+1; j1++ )
              {
                int j=j1+i2*(order+1)+i3*orderPow2;
                int k=j1+i2*(order+1)+i3*orderPow2;
                int l=j1+i2*(order+1)+j3*orderPow2;
                R[j]+= weights[j1]*weights[i2]*weights[i3]*( B[k][4]*dPhi[ i1 ][ j1 ]*dPhi[ i3 ][ j3 ] ) +
                       weights[j1]*weights[i2]*weights[j3]*( B[l][4]*dPhi[ i1 ][ j1 ]*dPhi[ j3 ][ i3 ] );
              }
            }
            // B23,B32 (B[][5])
            for( int j3=0; j3<order+1; j3++ )
            {
              for( int j2=0; j2<order+1; j2++ )
              {
                int j=i1+j2*(order+1)+j3*orderPow2;
                int k=i1+j2*(order+1)+i3*orderPow2;
                int l=i1+i2*(order+1)+j3*orderPow2;
                R[j]+= weights[i1]*weights[j2]*weights[i3]*(B[k][5]*dPhi[ i2 ][ i2 ]*dPhi[ i3 ][ j3 ]) +
                       weights[i1]*weights[i2]*weights[j3]*(B[l][5]*dPhi[ i2 ][ j2 ]*dPhi[ j3 ][ i3 ]);
              }
            }
    
            int i=i1+i2*(order+1)+i3*orderPow2;
            Y[i]=0;
            for( int j=0; j<nPointsPerElement; j++ )
            {
              Y[i]+=R[j]*pnLocal[j];
            }
  
          }
        }
      }
  }
  
  // compute stiffnessVector.
  // returns mass matrix and stiffness vector local to an element
  PROXY_HOST_DEVICE 
  static void computeMassMatrixAndStiffnessVector( const int & elementNumber,
                                                   const int & nPointsPerElement,
                                                   ARRAY_REAL_VIEW const & nodesCoordsX,
                                                   ARRAY_REAL_VIEW const & nodesCoordsY,
                                                   ARRAY_REAL_VIEW const & nodesCoordsZ,
                                                   PrecomputedData const & precomputedData,
                                                   float massMatrixLocal[],
                                                   float const pnLocal[],
                                                   float Y[])
  {
      float B[ROW][COL];
      float R[ROW];
      // compute Jacobian, massMatrix and B
      computeB( elementNumber, 
                ORDER, 
                precomputedData.weights, 
                nodesCoordsX,
                nodesCoordsY, 
                nodesCoordsZ, 
                precomputedData.derivativeBasisFunction1D, 
                massMatrixLocal, 
                B );
      // compute stifness  matrix ( durufle's optimization)
      gradPhiGradPhi( nPointsPerElement, 
                      ORDER, 
                      precomputedData.weights, 
                      precomputedData.derivativeBasisFunction1D, 
                      B, 
                      pnLocal,
                      R, 
                      Y );
  }
  
  /////////////////////////////////////////////////////////////////////////////////////
  //  end from first implementation
  /////////////////////////////////////////////////////////////////////////////////////
  
};
  
#endif //SEMQKGLINTEGRALSCLASSIC_HPP_
