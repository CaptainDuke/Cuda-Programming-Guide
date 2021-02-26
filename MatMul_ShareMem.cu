typedef struct {
    int width;
    int height;
    int stride;
    float* element;
} Matrix;

__device__ float GetElement(const Matrix A, int row, int col)
{
    float res = A.element[row * A.width + col];
    return res;
}

__device__ void SetElement(Matrix A, int row, int col, float value)
{
    A.element[row * A.width + col] = value;
}

__device__ Matrix GetSubMatrix(Matrix A, int row, int col)
{
    Matrix sub;
    sub.height = sub.width = BLOCK_SIZE;
    sub.stride = A.stride;
    sub.element = &A.element[A.width * BLOCK_SIZE * row
                                     + BLOCK_SIZE * col];
}

#define BLOCK_SIZE 16

__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);

void MatMul(const Matrix A, const Matrix B, Matrix C)
{
    Matrix d_A;
    d_A.width = A.width; d_A.height = A.height;
    d_A.stride = A.width;
    size_t size = d_A.width * d_A.height * sizeof(float);
    cudaMalloc(&d_A.element, size);
    cudaMemcpy(d_A.element, A.element, size, cudaMemcpyHostToDevic);

    Matrix d_B;
    d_B.width = B.width; d_B.height = B.height;
    d_B.stride = B.width;
    size = B.width * B.height * sizeof(float);
    cudaMalloc(&d_B.element, size);
    cudaMemcpy(d_B.element, B.element, size, cudaMemcpyHostToDevice);

    Matrix d_C;
    d_C.width = C.width; d_C.height = C.height;
    d_C.stride = C.stride;
    size = d_C.width * d_C.height * sizeof(float);
    cudaMalloc(&d_C.element, size);

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(B.width/dimBlock.x, A.height/dimBlock.y);
    MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

    cudaMemcpy(C.element, d_C.element, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A.element);
    cudaFree(d_B.element);
    cudaFree(d_C.element);


}

__global__ MatMulKernel(Matrix A, Matrix B, Matrix C)
{
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    int subRow = threadIdx.y;
    int subCol = threadIdx.x;

    Matrix Csub = GetSubMatrix(C, blockRow, blockCol);

    float Cval = 0;

    for(int i = 0; i < A.width/BLOCK_SIZE; i++)
    {
        Matrix Asub = GetSubMatrix(A, blockRow, i);
        Matrix Bsub = GetSubMatrix(B, i, blockCol);

        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        As[subRow][subCol] = GetElement(Asub, subRow, subCol);
        Bs[subRow][subCol] = GetElement(Bsub, subRow, subCol);

        __syncthreads();
        for(int j = 0; j < BLOCK_SIZE; j++)
        {
            Cval += As[subRow][j] * Bs[j][subCol];
        }
        __syncthreads();
    }
    SetElement(Csub, subRow, subCol, Cval);
}