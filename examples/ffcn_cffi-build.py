from cffi import FFI
ffibuilder = FFI()

# specify functions, etc made available to Python
ffibuilder.cdef("""
    void batch_fcn(double ans[], double x[], int n, int dim);
    """)

# specify code needed to build the module
ffibuilder.set_source(
    "ffcn_cffi",            # Python extension module (output)
    """
    // code for module
    double fcn_(double* x, int* dim);   // Fortran function in ffcn.o

    void batch_fcn(double ans[], double x[], int n, int dim)
    {
        int i;
        for(i=0; i<n; i++)
            ans[i] = fcn_(&x[i * dim], &dim);
    }
    """,
    extra_objects=['ffcn.o'],   # compiled Fortran
    libraries=['m'],            # may need to specify the math library (-lm)
    )

if __name__ == "__main__":
    # create C code for module and compile it
    ffibuilder.compile(verbose=False)
