#include<iostream>
#include<fstream>
#include<string>
#include<stdio.h>
#include<cuda.h>
#include<cuda_runtime_api.h>
#include<chrono>
using namespace std;

//--------------------------matrix.h-------------------------------------------------------------

typedef float Realna;

class matrix
{
    //string input;
    double *d;
    int dimenze;
    void alokujPamet(int dimenze);

public:
    matrix(string input);
    ~matrix();
    friend ostream& operator << (ostream &ost, const matrix &m);
    matrix inverze(matrix &m);
    void swap(int i, int j);
    //void vynasobKoef(Realna* d,double koef);
    void odecti(int i, int j, double koef); //vemu i a chci odecist od j
    void odectiInv(int i, int j, double koef);
    void identita();
    //void eliminace(int i,double koef);

};

//#endif // MATRIX_H

//---------------------------deklarace funkce.cu-------------------------------------------------


__global__ void Kernel(double* cuda_u, double* cuda_v, const int size, int slo);

__global__ void Vypis(double* cuda_u, const int size);

__global__ void Pivoting(double* cuda_u, double* cuda_v, const int size);


void odectiRadky(double* u, double* v, int velVektoru);



//-----------------------------main.cpp-----------------------------------------------------------

int main()
{
	cout <<"Jdeme na načtení matice"<<endl;
        //matrix a("prvniMatice.txt");
        //matrix a("msc01.txt");
        //matrix a("MyMatrix.txt");
        //matrix a("h.txt");
        matrix a("Velka.txt");

	cout<<"vypisu nactenou matici"<<endl;
    //cout << a << endl;
    cout << "nacteno"<< endl;
	cout<<"jdu startovat a.inverze"<<endl;
    matrix m = a.inverze(a);
    //cout << m << endl;
    return 0;
}

//--------------------------matrix.cpp---------------------------------------------------------------



matrix::matrix(string input)
{
    ifstream f(input);
    f >> dimenze;
    alokujPamet(dimenze);
    int row, col;

    for (int i = 0; i < dimenze; i++){
        row = (i/dimenze);
        col = (i%dimenze);
            f >> d[col + row * dimenze];
    }
}

void matrix::alokujPamet(int dimenze){

    d = new double[dimenze*dimenze];
}


ostream& operator <<(ostream &ost, const matrix &m){
        for (int i = 0; i<m.dimenze; i++){
            for(int j = 0; j<m.dimenze; j++){
                ost << m.d[i*m.dimenze + j] << '\t';
            }
            cout << '\n';
        }
    return ost;
}

matrix::~matrix(){
    delete d;
}



void matrix::identita(){
    int row, col;
    for(int i = 0;i<this->dimenze*this->dimenze;i++){
        row = i/this->dimenze;
        col = i%this->dimenze;
            if(row==col){this->d[row*this->dimenze + col]=1;}else{
                this->d[row*this->dimenze + col]=0;
            }
    }
}

matrix matrix::inverze(matrix &m){ 
 matrix inverze("Velka.txt");
inverze.identita();

	cout<<"jdu do funkce odectiRadky()"<<endl;
        odectiRadky(m.d,inverze.d,m.dimenze);

    return inverze;
} //konec inverze


//----------------------------------funkce.cu----------------------------------------------------------




void odectiRadky(double* u, double* v, int velVektoru){
    
    double *cuda_u, *cuda_v, *host_u, *host_v;
	host_u = new double[velVektoru*velVektoru];
	host_v = new double[velVektoru*velVektoru];
	for(int i = 0;i<velVektoru*velVektoru;i++){
		host_u[i] = u[i];
		host_v[i] = v[i];
		
	}
	
	const int size(velVektoru);

	

    if( cudaMalloc((void**) &cuda_u, size*size*sizeof(double)) != cudaSuccess ||
                  cudaMalloc((void**) &cuda_v, size*size*sizeof(double)) != cudaSuccess )

    {
        cerr  << "alokovani vektoru na GPU se nezdarilo" << endl;
    }

    if(cudaMemcpy((void* ) cuda_u, ( void* ) host_u,
                  size*size*sizeof ( double ), cudaMemcpyHostToDevice ) != cudaSuccess ||
            cudaMemcpy(( void* ) cuda_v, ( void* ) host_v,
                       size*size*sizeof( double ), cudaMemcpyHostToDevice ) != cudaSuccess )
    {
        cerr << "nepodarilo se zkopirovat vektory na GPU" << endl;
        
    }
    /**
      CUDA kernel
      */
        auto start = chrono::high_resolution_clock::now();
        dim3 cudaBlockSize(256);
        const int cudaBlocksCount = (size*size)/cudaBlockSize.x + ((size*size) % cudaBlockSize.x != 0);


        for(int i = 0;i<size;i++){
            Kernel<<<cudaBlocksCount,cudaBlockSize>>>(cuda_u,cuda_v,size,i);
            cudaError err = cudaGetLastError();
            if(err != cudaSuccess){
                cerr << "vypocet skoncil errorem: "<<cudaGetErrorString(err)<<" ."<< endl;
            }
            cudaDeviceSynchronize();
        }
        Pivoting<<<cudaBlocksCount,cudaBlockSize>>>(cuda_u,cuda_v,size);
        cudaError err2 = cudaGetLastError();
        if(err2 != cudaSuccess){
            cerr << "vypocet skoncil errorem: "<<cudaGetErrorString(err2)<<" 2."<< endl;
        }
        cudaDeviceSynchronize();

    auto finish = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = finish - start;
    cout << "elapsed: " << elapsed.count() << endl;
    //Kopirovani zpet z GPU
    if(cudaMemcpy( host_v, cuda_v, size*size*sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess)
    {
	cudaError err3 = cudaGetLastError();
        cerr << "nepodarilo se zkopirovat zpet z GPU, protoze: "<< cudaGetErrorString(err3) <<" ."<< endl;
    }
	
	
	for(int i = 0;i<velVektoru*velVektoru;i++){
		v[i] = host_v[i];
	}

	
	delete[] host_u;
	delete[] host_v;
	//delete size;
	cudaFree(cuda_u);
	cudaFree(cuda_v);

}



__global__ void Kernel(double* cuda_u, double* cuda_v, const int size, int slo){


int idx = threadIdx.x+blockIdx.x*blockDim.x;

if(idx>=size*size){
    return;}else{
    int row = idx/size;
    int col = idx % size;
    double pivot(cuda_u[slo+slo*size]);
    double koef(cuda_u[slo+row*size]/pivot);

    if(row!=slo){
        cuda_v[col+row*size] = cuda_v[col+row*size]-koef*cuda_v[size*slo+col];
        if(col>slo){
            cuda_u[col + size*row] = cuda_u[col + size*row]-koef*cuda_u[size*slo+col];
        }
    }else{
        return;}
    return;
}



} // KONEC KERNEL VOID


__global__ void Pivoting(double* cuda_u, double* cuda_v, const int size){
    int idx = threadIdx.x+blockIdx.x*blockDim.x;
    if(idx>=size*size){return;}else{
        int row = idx/size;
        int col = idx % size;
        cuda_v[col+row*size] = cuda_v[col+row*size]/cuda_u[row+row*size];
    }
}

__global__ void Vypis(double* cuda_u, const int size){

        for(int i = 0; i<size; i++){
            for(int j = 0; j<size;j++){
                printf(" %.2f ",cuda_u[j+i*size]);
            }
            printf("\n");
        }


}







